import type { Scene } from '../core/Scene';
import type { Camera } from '../core/Camera';
import type { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { Light } from '../lights/Light';
import { AmbientLight } from '../lights/AmbientLight';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { SpotLight } from '../lights/SpotLight';
import { SkinnedMesh } from '../core/SkinnedMesh';
import { InstancedMesh } from '../core/InstancedMesh';
import type { Material } from '../materials/Material';
import { StandardMaterial } from '../materials/StandardMaterial';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { ShaderMaterial, computeUniformLayout, packUniforms } from '../materials/ShaderMaterial';
import type { UniformLayout } from '../materials/ShaderMaterial';
import { Vector3 } from '../math/Vector3';
import { Matrix3 } from '../math/Matrix3';
import { Matrix4 } from '../math/Matrix4';
import { Frustum } from '../math/Frustum';
import { Sphere } from '../math/Sphere';
import { GeometryBuffers } from './GeometryBuffers';
import { TextureManager } from './TextureManager';
import { PipelineCache } from './PipelineCache';
import { DEPTH_FORMAT } from './constants';
import { PBR_SHADER, PBR_SKINNED_SHADER, PBR_INSTANCED_SHADER, PBR_MORPH_SHADER } from './shaders/pbr.wgsl';
import { LINE_SHADER } from './shaders/line.wgsl';
import { SHADOW_SHADER } from './shaders/shadow.wgsl';
import { ID_SHADER } from './shaders/id.wgsl';
import { SHADOW_DEPTH_FORMAT } from './constants';
import { PostProcessing } from './PostProcessing';
import { IBLPrefilter, IBL_MIP_LEVELS } from './IBLPrefilter';
import { CULL_SHADER } from './shaders/cull.wgsl';
import { CLUSTER_SHADER } from './shaders/clusters.wgsl';
import { buildSurfaceShader } from './shaders/surface.wgsl';

/** Either encoder accepts the same draw commands (pass or render bundle). */
type DrawEncoder = GPURenderPassEncoder | GPURenderBundleEncoder;

const MAX_LIGHTS = 256;
const UNIT_Y = new Vector3(0, 1, 0);
const UNIT_Z = new Vector3(0, 0, 1);
const FRAME_SIZE = 288; // bytes (view/proj/lightViewProj mat4s + cameraPos..clusterDims vec4s)
const MODEL_SIZE = 128;
const MATERIAL_SIZE = 144;
const LIGHT_STRIDE = 64; // bytes per light (positionKind + directionRange + colorDecay + spotParams)

// GPU-driven culling: indirect draw + compute-shader sphere cull.
const CULL_PARAMS_SIZE = 112;   // 6×vec4 planes (96) + u32 drawCount + 12 pad
const INDIRECT_STRIDE  = 20;    // GPUDrawIndexedIndirectParameters: 5 × u32

// Clustered forward+ grid (must match clusters.wgsl / pbr.wgsl).
const CLUSTER_X = 16;
const CLUSTER_Y = 9;
const CLUSTER_Z = 24;
const CLUSTER_COUNT = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const MAX_PER_CLUSTER = 32;
const CLUSTER_PARAMS_SIZE = 144; // view mat4 + invProj mat4 + info vec4

// Spot/point shadow atlas: 2048×2048 depth texture, 512×512 tiles (4×4 = 16 tiles).
// Tiles 0..MAX_SPOT_SHADOWS-1 hold spot lights; the rest hold point-light cube
// faces (6 consecutive tiles per point light).
const SPOT_ATLAS_SIZE = 2048;
const SPOT_TILE_SIZE = 512;
const MAX_SPOT_SHADOWS = 4;  // tiles reserved for shadow-casting spot lights
const MAX_POINT_SHADOWS = 2; // shadow-casting point lights (6 tiles each)
const MAX_SHADOW_TILES = MAX_SPOT_SHADOWS + MAX_POINT_SHADOWS * 6; // 16 = full atlas
const SHADOW_TILE_STRIDE = 80; // bytes: mat4x4 (64) + vec4 (16)

// Cube-face look directions/ups for point-light shadows (+X,-X,+Y,-Y,+Z,-Z) —
// the face order must match the dominant-axis selection in the PBR shader.
const POINT_FACE_DIRS = [
  new Vector3(1, 0, 0), new Vector3(-1, 0, 0),
  new Vector3(0, 1, 0), new Vector3(0, -1, 0),
  new Vector3(0, 0, 1), new Vector3(0, 0, -1),
];
const POINT_FACE_UPS = [UNIT_Y, UNIT_Y, UNIT_Z, UNIT_Z, UNIT_Y, UNIT_Y];

// TAA: 8-sample Halton(2,3) sub-pixel jitter pattern.
const TAA_SAMPLES = 8;

/** Radical-inverse Halton sequence value in [0, 1) for the given 1-based index. */
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

interface SkinnedResources {
  boneBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  jointCount: number;
}

interface InstancedResources {
  buffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  count: number;
  version: number;
}

interface MaterialResources {
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  textureSignature: string;
}

interface MorphResources {
  infoBuffer: GPUBuffer;
  weightBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  count: number;
}

interface ShaderMaterialResources {
  buffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  data: Float32Array<ArrayBuffer>;
  layout: UniformLayout;
  shapeKey: string;       // uniform names+kinds; a change rebuilds buffer + shader
  uploadedFrame: number;  // last frameNumber the uniform values were written
}

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  /** Device pixel ratio cap. Defaults to window.devicePixelRatio. */
  pixelRatio?: number;
  /** Multisample count (1 or 4). Defaults to 4. */
  sampleCount?: number;
  powerPreference?: GPUPowerPreference;
}

/** A WebGPU forward renderer with clustered-free multi-light PBR shading. */
export class WebGPURenderer {
  canvas: HTMLCanvasElement;
  device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private sampleCount: number;
  private pixelRatio: number;

  private geometries!: GeometryBuffers;
  private textures!: TextureManager;
  private pipelines!: PipelineCache;

  private frameBuffer!: GPUBuffer;
  private lightBuffer!: GPUBuffer;
  private frameBindGroup!: GPUBindGroup;
  private frameData = new Float32Array(FRAME_SIZE / 4);
  private lightData = new Float32Array((MAX_LIGHTS * LIGHT_STRIDE) / 4);

  private depthTexture!: GPUTexture;
  private depthSampleView: GPUTextureView | null = null; // depth-only view for SSAO sampling
  private msaaTexture: GPUTexture | null = null;
  private width = 1;
  private height = 1;

  // Shadow mapping (directional, single caster).
  /** Enable directional shadow mapping for the first `castShadow` light. */
  shadows = false;
  /** Shadow map resolution (square). */
  shadowMapSize = 2048;
  /** World-space normal offset applied before the depth comparison (reduces acne). */
  shadowNormalBias = 0.02;
  private shadowTexture!: GPUTexture;
  private shadowView!: GPUTextureView;
  private shadowSampler!: GPUSampler;
  private shadowLightBuffer!: GPUBuffer;
  private shadowLightBindGroup!: GPUBindGroup;
  private shadowMapAllocated = 0;
  // Environment (IBL): resolved each frame from scene.environment.
  private envView!: GPUTextureView;
  private envSampler!: GPUSampler;
  private envEnabled = false;
  private envIntensity = 1;
  private envMaxMip = 0;
  private envKey = '';
  private ibl!: IBLPrefilter;
  private iblActive = false; // true once convolve() has run for the current env
  private iblEnvKey = '';   // tracks which env was last convolved

  // Post-processing (opt-in): render to an HDR target, then tonemap (+ FXAA).
  /** Route rendering through the HDR post pipeline (tonemap moves to a final pass). */
  postProcessing = false;
  /** Apply FXAA in the post pipeline (only when `postProcessing` is on). */
  fxaa = true;
  /** Apply bloom in the post pipeline (only when `postProcessing` is on). */
  bloom = false;
  bloomThreshold = 1.0;
  bloomIntensity = 0.6;
  /**
   * Order-independent transparency (weighted-blended). Requires `postProcessing`
   * and sampleCount 1; otherwise transparent meshes fall back to sorted blending.
   */
  oit = false;
  /**
   * Screen-space ambient occlusion. Requires `postProcessing = true` and
   * `sampleCount = 1`; otherwise silently disabled.
   */
  ssao = false;
  ssaoRadius = 0.5;
  ssaoBias = 0.025;
  ssaoStrength = 1.0;
  /**
   * Temporal anti-aliasing: a sub-pixel Halton jitter is baked into the
   * projection matrix and a resolve pass blends the reprojected history
   * (camera-motion only). Requires `postProcessing = true` and `sampleCount = 1`;
   * otherwise silently disabled.
   */
  taa = false;
  /** TAA blend factor: weight of the current frame (lower = smoother, more ghosting). */
  taaBlend = 0.1;
  private taaActive = false;     // was TAA running last frame (history validity)
  private taaFrameIndex = 0;     // Halton sequence cursor
  private _jitterX = 0;          // NDC jitter baked into this frame's projection
  private _jitterY = 0;
  private _prevViewProj = new Float32Array(16);
  private _invViewProj = new Matrix4();
  /** Record the opaque draws into a render bundle to amortize encoding cost. */
  renderBundles = false;
  private opaqueBundle: GPURenderBundle | null = null;
  /**
   * GPU-driven frustum culling: a compute shader tests bounding spheres against
   * the camera frustum and writes instanceCount 0/1 into an indirect draw buffer.
   * Opaque indexed meshes use `drawIndexedIndirect`; instanced and non-indexed
   * meshes always follow the normal CPU path.  Incompatible with `renderBundles`.
   */
  gpuCulling = false;
  /**
   * Clustered forward+ lighting: a compute shader bins light bounding spheres
   * into a 16×9×24 screen-tile/depth-slice grid each frame; fragments shade
   * only the lights in their cluster (up to 32), so hundreds of ranged lights
   * stay cheap. Perspective cameras only; ranged point/spot lights benefit —
   * directional and infinite-range lights still hit every cluster.
   */
  clusteredLighting = false;
  private clusterPipeline: GPUComputePipeline | null = null;
  private clusterLayout: GPUBindGroupLayout | null = null;
  private clusterParamsBuffer: GPUBuffer | null = null;
  private clusterLightsBuffer: GPUBuffer | null = null;
  private clusterDummyBuffer: GPUBuffer | null = null;
  private clusterBindGroup: GPUBindGroup | null = null;
  private gpuCullPipeline: GPUComputePipeline | null = null;
  private gpuCullLayout: GPUBindGroupLayout | null = null;
  private gpuCullParamsBuffer: GPUBuffer | null = null;
  private gpuSphereBuffer: GPUBuffer | null = null;
  private gpuIndirectBuffer: GPUBuffer | null = null;
  private gpuCullBindGroup: GPUBindGroup | null = null;
  private gpuCullCapacity = 0;
  private bundleKey = '';
  private frameBindGroupVersion = 0;
  private post!: PostProcessing;
  private _lightView = new Matrix4();
  private _lightProj = new Matrix4();
  private _lightViewProj = new Matrix4();
  private _lightPos = new Vector3();
  private _lightDir = new Vector3();
  private _sceneCenter = new Vector3();
  private _corner = new Vector3();
  private shadowCasterIndex = -1;

  // Spot-light shadow atlas resources.
  private spotAtlasTexture: GPUTexture | null = null;
  private spotAtlasView: GPUTextureView | null = null;
  private spotAtlasSampler: GPUSampler | null = null;
  private spotShadowTilesBuffer: GPUBuffer | null = null;
  private spotLightBuffers: GPUBuffer[] = [];
  private spotLightBindGroups: GPUBindGroup[] = [];
  private spotAtlasAllocated = 0; // 0 = not allocated, SPOT_ATLAS_SIZE = allocated full
  // Populated by prepareShadow(); read by uploadFrame() and renderSpotShadowPasses().
  private spotShadowCasters: Array<{
    tileIndex: number;
    packedIndex: number;
    viewProj: Matrix4;
  }> = [];

  // Single pool buffer for all mesh model matrices (dynamic-offset uniform).
  private modelPoolBuffer: GPUBuffer | null = null;
  private modelPoolBindGroup: GPUBindGroup | null = null;
  private modelPoolCapacity = 0;
  private meshSlots = new WeakMap<Mesh, number>();
  private nextMeshSlot = 0;

  private materialResources = new WeakMap<Material, MaterialResources>();
  private shaderMaterialResources = new WeakMap<ShaderMaterial, ShaderMaterialResources>();
  // Skybox: frame uniform + raw env view (rebuilt when the environment changes).
  private skyBindGroup: GPUBindGroup | null = null;
  private skyBindGroupKey = '';
  // Scene color format for this frame's pipelines (HDR under post-processing).
  private sceneTargetFormat: GPUTextureFormat = 'bgra8unorm';
  private frameNumber = 0;
  private readonly clockStart = performance.now();
  private skinnedResources = new WeakMap<SkinnedMesh, SkinnedResources>();
  private instancedResources = new WeakMap<InstancedMesh, InstancedResources>();
  private morphResources = new WeakMap<Mesh, MorphResources>();
  private lineResources = new WeakMap<LineBasicMaterial, { buffer: GPUBuffer; bindGroup: GPUBindGroup }>();

  // Id-buffer picking (all lazy — created on first pickAt call)
  private idPipeline: GPURenderPipeline | null = null;
  private idLayout: GPUBindGroupLayout | null = null;
  private idUniformBuffer: GPUBuffer | null = null;
  private idBindGroup: GPUBindGroup | null = null;
  private idBufferCapacity = 0;
  private idColorTexture: GPUTexture | null = null;
  private idDepthTexture: GPUTexture | null = null;
  private idReadBuffer: GPUBuffer | null = null;
  private idTexW = 0;
  private idTexH = 0;

  private _normalMatrix = new Matrix3();
  private _camPos = new Vector3();
  private _meshPos = new Vector3();
  private _viewProjection = new Matrix4();
  private _frustum = new Frustum();
  private _worldSphere = new Sphere();

  /** Skip meshes whose bounding sphere is outside the camera frustum. */
  frustumCulling = true;
  /** Number of meshes culled in the last frame (diagnostics). */
  culledCount = 0;

  // scratch render lists
  private opaque: Mesh[] = [];
  private transparent: Mesh[] = [];
  private lights: Light[] = [];

  constructor(options: RendererOptions) {
    this.canvas = options.canvas;
    this.sampleCount = options.sampleCount ?? 4;
    this.pixelRatio = options.pixelRatio ?? Math.min(window.devicePixelRatio || 1, 2);
  }

  static isSupported(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  async init(): Promise<void> {
    if (!WebGPURenderer.isSupported()) {
      throw new Error('WebGPU is not available in this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No suitable GPUAdapter found.');
    this.device = await adapter.requestDevice();
    this.device.lost.then((info) => {
      console.error('[vela] WebGPU device lost:', info.message);
    });

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    this.geometries = new GeometryBuffers(this.device);
    this.textures = new TextureManager(this.device);
    this.pipelines = new PipelineCache(
      this.device, this.format, this.sampleCount,
      PBR_SHADER, PBR_SKINNED_SHADER, PBR_INSTANCED_SHADER, PBR_MORPH_SHADER, LINE_SHADER, SHADOW_SHADER,
    );

    this.createShadowResources();
    this.envView = this.textures.defaultWhiteView;
    this.envSampler = this.textures.defaultSampler;
    this.post = new PostProcessing(this.device, this.format, this.sampleCount);
    this.ibl = new IBLPrefilter(this.device);
    // Compute the BRDF LUT once at startup.
    const brdfEncoder = this.device.createCommandEncoder();
    this.ibl.computeBRDFLUT(brdfEncoder);
    this.device.queue.submit([brdfEncoder.finish()]);
    this.createFrameResources();
    this.setSize(this.canvas.clientWidth || 800, this.canvas.clientHeight || 600);
  }

  private createFrameResources(): void {
    this.frameBuffer = this.device.createBuffer({
      size: FRAME_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.lightBuffer = this.device.createBuffer({
      size: MAX_LIGHTS * LIGHT_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Placeholder for the cluster-light list until clustered lighting is enabled.
    this.clusterDummyBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.buildFrameBindGroup();
  }

  private buildFrameBindGroup(): void {
    this.frameBindGroupVersion++; // invalidates any recorded render bundle
    const iblEnvView = this.iblActive ? this.ibl.prefilteredView : this.textures.defaultWhiteView;
    const iblSampler = this.iblActive ? this.ibl.sampler : this.textures.defaultSampler;
    this.frameBindGroup = this.device.createBindGroup({
      layout: this.pipelines.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: this.shadowView },
        { binding: 3, resource: this.shadowSampler },
        // binding 4: specular env (prefiltered when IBL active, raw env otherwise)
        { binding: 4, resource: this.iblActive ? iblEnvView : this.envView },
        { binding: 5, resource: this.iblActive ? iblSampler : this.envSampler },
        // bindings 6-9: IBL irradiance + BRDF LUT (defaults when IBL not active)
        { binding: 6, resource: this.iblActive ? this.ibl.irradianceView : this.textures.defaultWhiteView },
        { binding: 7, resource: iblSampler },
        { binding: 8, resource: this.iblActive ? this.ibl.brdfLUTView : this.textures.defaultWhiteView },
        { binding: 9, resource: iblSampler },
        // bindings 10-12: spot-light shadow atlas (dummy when not allocated)
        { binding: 10, resource: { buffer: this.spotShadowTilesBuffer! } },
        { binding: 11, resource: this.spotAtlasView! },
        { binding: 12, resource: this.spotAtlasSampler! },
        // binding 13: screen-space refraction capture (post.sceneCaptureView is always valid)
        { binding: 13, resource: this.post.sceneCaptureView },
        // binding 14: clustered light lists (dummy until clustered lighting runs)
        { binding: 14, resource: { buffer: this.clusterLightsBuffer ?? this.clusterDummyBuffer! } },
      ],
    });
  }

  /** Allocate directional shadow map + spot atlas (1×1 dummies when shadows disabled). */
  private createShadowResources(): void {
    const size = this.shadows ? this.shadowMapSize : 1;
    if (this.shadowMapAllocated !== size) {
      this.shadowTexture?.destroy();
      this.shadowTexture = this.device.createTexture({
        label: 'shadow-map',
        size: [size, size],
        format: SHADOW_DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.shadowView = this.shadowTexture.createView();
      this.shadowMapAllocated = size;
      if (this.frameBuffer) this.buildFrameBindGroup(); // shadowView changed → rebind
    }
    if (!this.shadowSampler) {
      this.shadowSampler = this.device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    }
    if (!this.shadowLightBuffer) {
      this.shadowLightBuffer = this.device.createBuffer({
        size: 64, // lightViewProj mat4
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.shadowLightBindGroup = this.device.createBindGroup({
        layout: this.pipelines.shadowLightLayout,
        entries: [{ binding: 0, resource: { buffer: this.shadowLightBuffer } }],
      });
    }

    // Spot-light shadow atlas (lazily allocated on first use; 1×1 dummy until then).
    const atlasSize = this.shadows ? SPOT_ATLAS_SIZE : 1;
    if (this.spotAtlasAllocated !== atlasSize) {
      this.spotAtlasTexture?.destroy();
      this.spotAtlasTexture = this.device.createTexture({
        label: 'spot-shadow-atlas',
        size: [atlasSize, atlasSize],
        format: SHADOW_DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.spotAtlasView = this.spotAtlasTexture.createView();
      this.spotAtlasAllocated = atlasSize;
      if (this.frameBuffer) this.buildFrameBindGroup();
    }
    if (!this.spotAtlasSampler) {
      this.spotAtlasSampler = this.device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    }
    if (!this.spotShadowTilesBuffer) {
      this.spotShadowTilesBuffer = this.device.createBuffer({
        size: MAX_SHADOW_TILES * SHADOW_TILE_STRIDE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    if (this.spotLightBuffers.length === 0) {
      for (let i = 0; i < MAX_SHADOW_TILES; i++) {
        const buf = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bg = this.device.createBindGroup({
          layout: this.pipelines.shadowLightLayout,
          entries: [{ binding: 0, resource: { buffer: buf } }],
        });
        this.spotLightBuffers.push(buf);
        this.spotLightBindGroups.push(bg);
      }
    }
  }

  setPixelRatio(ratio: number): void {
    this.pixelRatio = ratio;
    this.setSize(this.width, this.height);
  }

  /** Resize backing store. width/height are CSS pixels. */
  setSize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    const w = Math.max(1, Math.floor(width * this.pixelRatio));
    const h = Math.max(1, Math.floor(height * this.pixelRatio));
    this.canvas.width = w;
    this.canvas.height = h;

    this.depthTexture?.destroy();
    // sampleCount=1 depth textures also expose TEXTURE_BINDING for SSAO sampling.
    const depthUsage = GPUTextureUsage.RENDER_ATTACHMENT |
      (this.sampleCount === 1 ? GPUTextureUsage.TEXTURE_BINDING : 0);
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: DEPTH_FORMAT,
      sampleCount: this.sampleCount,
      usage: depthUsage,
    });
    this.depthSampleView = this.sampleCount === 1
      ? this.depthTexture.createView({ aspect: 'depth-only' })
      : null;

    this.msaaTexture?.destroy();
    this.msaaTexture = null;
    if (this.sampleCount > 1) {
      this.msaaTexture = this.device.createTexture({
        size: [w, h],
        format: this.format,
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
  }

  get drawingBufferWidth(): number {
    return this.canvas.width;
  }
  get drawingBufferHeight(): number {
    return this.canvas.height;
  }

  render(scene: Scene, camera: Camera): void {
    scene.updateMatrixWorld();
    camera.updateMatrixWorld();
    this.frameNumber++;
    // The scene pass renders into the HDR target when post-processing is on,
    // so material pipelines must target that format.
    this.sceneTargetFormat = this.postProcessing ? 'rgba16float' : this.format;

    // Build the view frustum for culling (projection * view).
    this._viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._viewProjection);

    // TAA: pick this frame's sub-pixel jitter (baked into the uploaded projection).
    const useTAA = this.taa && this.postProcessing && this.sampleCount === 1 && !!this.depthSampleView;
    if (useTAA) {
      const i = (this.taaFrameIndex++ % TAA_SAMPLES) + 1;
      this._jitterX = (halton(i, 2) * 2 - 1) / this.canvas.width;
      this._jitterY = (halton(i, 3) * 2 - 1) / this.canvas.height;
      if (!this.taaActive) this.post?.invalidateTAAHistory(); // just (re)enabled
    } else {
      this._jitterX = 0;
      this._jitterY = 0;
    }
    this.taaActive = useTAA;

    this.collect(scene);
    this.prepareShadow();
    this.uploadFrame(scene, camera);

    // Sort transparent back-to-front
    camera.getWorldPosition(this._camPos);
    this.transparent.sort((a, b) => {
      const da = a.getWorldPosition(this._meshPos).distanceToSquared(this._camPos);
      this._meshPos.set(0, 0, 0);
      const db = b.getWorldPosition(this._meshPos).distanceToSquared(this._camPos);
      return db - da;
    });

    const encoder = this.device.createCommandEncoder();

    // IBL prefilter compute passes run first (before any render pass).
    this.prepareEnvironment(scene, encoder);

    // Shadow depth passes (before the main pass).
    if (this.shadowCasterIndex >= 0) this.renderShadowPass(encoder);
    if (this.spotShadowCasters.length > 0) this.renderSpotShadowPasses(encoder);
    // GPU frustum cull: fills instanceCount in the indirect buffer.
    this.prepareGpuCull(encoder);
    // Clustered forward+: bin lights into the cluster grid.
    this.prepareClusters(encoder, camera);

    const swapView = this.context.getCurrentTexture().createView();
    const clear = this.clearColor(scene);

    let colorAttachment: GPURenderPassColorAttachment;
    if (this.postProcessing) {
      this.post.ensureSize(this.canvas.width, this.canvas.height);
      colorAttachment = this.post.sceneColorAttachment(clear);
    } else if (this.sampleCount > 1) {
      colorAttachment = {
        view: this.msaaTexture!.createView(),
        resolveTarget: swapView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: clear,
      };
    } else {
      colorAttachment = { view: swapView, loadOp: 'clear', storeOp: 'store', clearValue: clear };
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setBindGroup(0, this.frameBindGroup);

    if (this.renderBundles && this.opaque.length) {
      this.refreshOpaqueUniforms(); // keep per-object buffers fresh for the replay
      pass.executeBundles([this.getOpaqueBundle()]);
    } else {
      for (const mesh of this.opaque) this.drawMesh(pass, mesh);
    }

    // Skybox: draws the environment where depth is still 1 (after opaques, so
    // covered pixels are rejected; before transparents, so it sits behind them).
    if (scene.skybox && this.envEnabled) {
      if (!this.skyBindGroup || this.skyBindGroupKey !== this.envKey) {
        this.skyBindGroup = this.device.createBindGroup({
          layout: this.pipelines.skyLayout,
          entries: [
            { binding: 0, resource: { buffer: this.frameBuffer } },
            { binding: 1, resource: this.envView },
            { binding: 2, resource: this.envSampler },
          ],
        });
        this.skyBindGroupKey = this.envKey;
      }
      pass.setPipeline(this.pipelines.getSky(this.sceneTargetFormat));
      pass.setBindGroup(0, this.skyBindGroup);
      pass.draw(3);
      pass.setBindGroup(0, this.frameBindGroup); // restore for transparent draws
    }

    const useOIT = this.oit && this.postProcessing && this.transparent.length > 0;
    if (!useOIT) {
      for (const mesh of this.transparent) this.drawMesh(pass, mesh);
    }

    pass.end();

    // Capture the opaque HDR scene for screen-space refraction before transparent draws.
    if (this.postProcessing && this.transparent.length > 0) {
      this.post.captureHDR(encoder);
    }

    // Order-independent transparency: accumulate into HDR-side targets, then composite.
    if (useOIT) {
      const oitPass = encoder.beginRenderPass({
        colorAttachments: this.post.oitColorAttachments(),
        depthStencilAttachment: { view: this.depthTexture.createView(), depthReadOnly: true },
      });
      oitPass.setBindGroup(0, this.frameBindGroup);
      for (const mesh of this.transparent) this.drawMesh(oitPass, mesh, true);
      oitPass.end();
      this.post.compositeOIT(encoder);
    }

    const useSSAO = this.ssao && this.postProcessing && this.sampleCount === 1 && !!this.depthSampleView;
    // Resolve the HDR target through the post chain into the swap chain.
    if (this.postProcessing) {
      if (useSSAO) {
        this.post.runSSAO(
          encoder,
          this.depthSampleView!,
          new Float32Array(camera.projectionMatrixInverse.elements),
          new Float32Array(camera.projectionMatrix.elements),
          this.ssaoRadius,
          this.ssaoBias,
        );
      }
      // TAA resolve: blend reprojected history into a ping-pong target that
      // replaces the HDR view as the post-chain input.
      let postInput: GPUTextureView | undefined;
      if (useTAA) {
        this._invViewProj.copy(this._viewProjection).invert();
        postInput = this.post.runTAA(
          encoder,
          this.depthSampleView!,
          this._prevViewProj,
          new Float32Array(this._invViewProj.elements),
          this._jitterX,
          this._jitterY,
          this.taaBlend,
        );
      }
      this.post.run(encoder, swapView, {
        fxaa: this.fxaa,
        bloom: this.bloom,
        bloomThreshold: this.bloomThreshold,
        bloomIntensity: this.bloomIntensity,
        ssao: useSSAO,
        ssaoStrength: this.ssaoStrength,
      }, postInput);
    }
    // Save this frame's unjittered view-projection for next frame's reprojection.
    this._prevViewProj.set(this._viewProjection.elements);

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Choose the shadow-casting directional light and fit an orthographic light
   * frustum to the opaque scene bounds, producing `_lightViewProj`. Sets
   * `shadowCasterIndex` (the light's index in the packed light array) or -1.
   */
  private prepareShadow(): void {
    this.shadowCasterIndex = -1;
    this.spotShadowCasters.length = 0;
    if (!this.shadows) return;
    this.createShadowResources(); // (re)allocate if size/enabled changed

    // Find the first directional light flagged castShadow, tracking its packed index.
    let caster: DirectionalLight | null = null;
    let packedIndex = -1;
    let spotCount = 0;
    let pointCount = 0;
    let i = 0;
    for (const light of this.lights) {
      if (light instanceof AmbientLight) continue;
      if (light instanceof DirectionalLight && light.castShadow && !caster) {
        caster = light;
        packedIndex = i;
      } else if (light instanceof SpotLight && light.castShadow && spotCount < MAX_SPOT_SHADOWS) {
        this.prepareSpotShadow(light, i, spotCount);
        spotCount++;
      } else if (light instanceof PointLight && light.castShadow && pointCount < MAX_POINT_SHADOWS) {
        this.preparePointShadow(light, i, pointCount);
        pointCount++;
      }
      i++;
    }
    if (!caster && this.spotShadowCasters.length === 0) return;

    if (caster) {
      // World-space bounds of the opaque casters (from their bounding spheres).
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const mesh of this.opaque) {
        const geo = mesh.geometry;
        if (!geo.boundingSphere) geo.computeBoundingSphere();
        const s = geo.boundingSphere;
        if (!s || s.isEmpty()) continue;
        this._worldSphere.copy(s).applyMatrix4(mesh.matrixWorld);
        const c = this._worldSphere.center, r = this._worldSphere.radius;
        minX = Math.min(minX, c.x - r); maxX = Math.max(maxX, c.x + r);
        minY = Math.min(minY, c.y - r); maxY = Math.max(maxY, c.y + r);
        minZ = Math.min(minZ, c.z - r); maxZ = Math.max(maxZ, c.z + r);
      }
      if (minX <= maxX) {
        this._sceneCenter.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
        const radius = 0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);

        caster.getWorldPosition(this._lightPos);
        caster.target.updateWorldMatrix(true, false);
        this._lightDir.set(
          caster.target.matrixWorld.elements[12] - this._lightPos.x,
          caster.target.matrixWorld.elements[13] - this._lightPos.y,
          caster.target.matrixWorld.elements[14] - this._lightPos.z,
        );
        if (this._lightDir.lengthSq() === 0) this._lightDir.set(0, -1, 0);
        this._lightDir.normalize();

        const eye = this._lightPos.set(
          this._sceneCenter.x - this._lightDir.x * radius,
          this._sceneCenter.y - this._lightDir.y * radius,
          this._sceneCenter.z - this._lightDir.z * radius,
        );
        const up = Math.abs(this._lightDir.y) > 0.99 ? UNIT_Z : UNIT_Y;
        this._lightView.identity().lookAt(eye, this._sceneCenter, up).setPosition(eye).invert();

        let lminX = Infinity, lminY = Infinity, lminZ = Infinity;
        let lmaxX = -Infinity, lmaxY = -Infinity, lmaxZ = -Infinity;
        for (let k = 0; k < 8; k++) {
          this._corner.set(k & 1 ? maxX : minX, k & 2 ? maxY : minY, k & 4 ? maxZ : minZ);
          this._corner.applyMatrix4(this._lightView);
          lminX = Math.min(lminX, this._corner.x); lmaxX = Math.max(lmaxX, this._corner.x);
          lminY = Math.min(lminY, this._corner.y); lmaxY = Math.max(lmaxY, this._corner.y);
          lminZ = Math.min(lminZ, this._corner.z); lmaxZ = Math.max(lmaxZ, this._corner.z);
        }
        const near = -lmaxZ;
        const far = -lminZ;
        this._lightProj.makeOrthographic(lminX, lmaxX, lmaxY, lminY, near, far);
        this._lightViewProj.multiplyMatrices(this._lightProj, this._lightView);

        this.device.queue.writeBuffer(this.shadowLightBuffer, 0, new Float32Array(this._lightViewProj.elements));
        this.shadowCasterIndex = packedIndex;
      }
    }

    // Upload shadow tile data (viewProj + atlas region) to the storage buffer.
    if (this.spotShadowCasters.length > 0 && this.spotShadowTilesBuffer) {
      const tileData = new Float32Array(MAX_SHADOW_TILES * (SHADOW_TILE_STRIDE / 4));
      const tilesPerRow = SPOT_ATLAS_SIZE / SPOT_TILE_SIZE;
      for (const sc of this.spotShadowCasters) {
        const base = sc.tileIndex * (SHADOW_TILE_STRIDE / 4);
        tileData.set(sc.viewProj.elements, base);
        const col = sc.tileIndex % tilesPerRow;
        const row = Math.floor(sc.tileIndex / tilesPerRow);
        tileData[base + 16] = (col * SPOT_TILE_SIZE) / SPOT_ATLAS_SIZE; // uv offset x
        tileData[base + 17] = (row * SPOT_TILE_SIZE) / SPOT_ATLAS_SIZE; // uv offset y
        tileData[base + 18] = SPOT_TILE_SIZE / SPOT_ATLAS_SIZE;          // uv scale
        tileData[base + 19] = 1 / SPOT_ATLAS_SIZE;                       // texel step
      }
      this.device.queue.writeBuffer(this.spotShadowTilesBuffer, 0, tileData);
    }
  }

  /** Compute the perspective viewProj for a shadow-casting SpotLight and record it. */
  private prepareSpotShadow(light: SpotLight, packedIndex: number, tileIndex: number): void {
    light.getWorldPosition(this._lightPos);
    light.target.updateWorldMatrix(true, false);
    this._lightDir.set(
      light.target.matrixWorld.elements[12] - this._lightPos.x,
      light.target.matrixWorld.elements[13] - this._lightPos.y,
      light.target.matrixWorld.elements[14] - this._lightPos.z,
    );
    if (this._lightDir.lengthSq() === 0) this._lightDir.set(0, 0, -1);
    this._lightDir.normalize();

    const up = Math.abs(this._lightDir.y) > 0.99 ? UNIT_Z : UNIT_Y;
    const target = this._lightPos.clone().add(this._lightDir);
    this._lightView.identity().lookAt(this._lightPos, target, up).setPosition(this._lightPos).invert();
    const far = light.distance > 0 ? light.distance : 200;
    this._lightProj.makePerspective(light.angle * 2, 1, 0.05, far);
    const vp = new Matrix4().multiplyMatrices(this._lightProj, this._lightView);

    // Write to the per-tile uniform buffer (separate buffer per tile = no writeBuffer conflict).
    this.device.queue.writeBuffer(this.spotLightBuffers[tileIndex], 0, new Float32Array(vp.elements));

    this.spotShadowCasters.push({ tileIndex, packedIndex, viewProj: vp });
  }

  /**
   * Record 6 cube-face shadow tiles (90° perspective each) for a shadow-casting
   * PointLight. Faces occupy consecutive atlas tiles starting after the spot
   * tiles; the PBR shader picks the face from the dominant axis of the
   * light→fragment vector.
   */
  private preparePointShadow(light: PointLight, packedIndex: number, pointIndex: number): void {
    light.getWorldPosition(this._lightPos);
    const far = light.distance > 0 ? light.distance : 200;
    this._lightProj.makePerspective(Math.PI / 2, 1, 0.05, far);
    for (let face = 0; face < 6; face++) {
      const tileIndex = MAX_SPOT_SHADOWS + pointIndex * 6 + face;
      const target = this._lightPos.clone().add(POINT_FACE_DIRS[face]);
      this._lightView.identity().lookAt(this._lightPos, target, POINT_FACE_UPS[face]).setPosition(this._lightPos).invert();
      const vp = new Matrix4().multiplyMatrices(this._lightProj, this._lightView);
      this.device.queue.writeBuffer(this.spotLightBuffers[tileIndex], 0, new Float32Array(vp.elements));
      this.spotShadowCasters.push({ tileIndex, packedIndex, viewProj: vp });
    }
  }

  /** Resolve `scene.environment` into the env bindings, rebuilding on change. */
  private prepareEnvironment(scene: Scene, encoder: GPUCommandEncoder): void {
    const env = scene.environment;
    if (env && env.source) {
      const entry = this.textures.get(env);
      const key = `${env.id}:${env.version}`;
      if (key !== this.envKey) {
        this.envView = entry.view;
        this.envSampler = entry.sampler;
        this.envKey = key;
        // Trigger IBL prefilter whenever the environment texture changes.
        if (key !== this.iblEnvKey) {
          this.ibl.convolve(encoder, entry.view);
          this.iblEnvKey = key;
          this.iblActive = true;
        }
        this.buildFrameBindGroup();
      }
      this.envEnabled = true;
      this.envIntensity = scene.environmentIntensity;
      // When IBL is active the specular map has IBL_MIP_LEVELS mip levels.
      this.envMaxMip = this.iblActive ? IBL_MIP_LEVELS - 1 : Math.max(0, entry.texture.mipLevelCount - 1);
    } else {
      if (this.envKey !== '') {
        this.envView = this.textures.defaultWhiteView;
        this.envSampler = this.textures.defaultSampler;
        this.envKey = '';
        this.iblActive = false;
        this.buildFrameBindGroup();
      }
      this.envEnabled = false;
    }
  }

  private renderShadowPass(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.pipelines.shadowPipeline);
    pass.setBindGroup(0, this.shadowLightBindGroup);
    for (const mesh of this.opaque) {
      if (mesh instanceof InstancedMesh) continue; // instanced casters unsupported in v1
      const geometry = this.geometries.get(mesh.geometry);
      pass.setBindGroup(1, this.modelPoolBindGroup!, [this.getMeshSlot(mesh) * 256]);
      pass.setVertexBuffer(0, geometry.position);
      if (geometry.index) {
        pass.setIndexBuffer(geometry.index, geometry.indexFormat);
        pass.drawIndexed(geometry.drawCount);
      } else {
        pass.draw(geometry.drawCount);
      }
    }
    pass.end();
  }

  private renderSpotShadowPasses(encoder: GPUCommandEncoder): void {
    const ts = SPOT_TILE_SIZE;
    const tilesPerRow = SPOT_ATLAS_SIZE / SPOT_TILE_SIZE;
    for (let ci = 0; ci < this.spotShadowCasters.length; ci++) {
      const sc = this.spotShadowCasters[ci];
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.spotAtlasView!,
          depthClearValue: 1.0,
          depthLoadOp: ci === 0 ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      });
      const col = sc.tileIndex % tilesPerRow;
      const row = Math.floor(sc.tileIndex / tilesPerRow);
      pass.setViewport(col * ts, row * ts, ts, ts, 0, 1);
      pass.setScissorRect(col * ts, row * ts, ts, ts);
      pass.setPipeline(this.pipelines.shadowPipeline);
      pass.setBindGroup(0, this.spotLightBindGroups[sc.tileIndex]);
      for (const mesh of this.opaque) {
        if (mesh instanceof InstancedMesh) continue;
        const geometry = this.geometries.get(mesh.geometry);
        pass.setBindGroup(1, this.modelPoolBindGroup!, [this.getMeshSlot(mesh) * 256]);
        pass.setVertexBuffer(0, geometry.position);
        if (geometry.index) {
          pass.setIndexBuffer(geometry.index, geometry.indexFormat);
          pass.drawIndexed(geometry.drawCount);
        } else {
          pass.draw(geometry.drawCount);
        }
      }
      pass.end();
    }
  }

  private ensureClusterResources(): void {
    if (this.clusterPipeline) return;
    const module = this.device.createShaderModule({ code: CLUSTER_SHADER, label: 'clusters' });
    this.clusterLayout = this.device.createBindGroupLayout({
      label: 'clusters',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.clusterPipeline = this.device.createComputePipeline({
      label: 'clusters',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.clusterLayout] }),
      compute: { module, entryPoint: 'cs_cluster' },
    });
    this.clusterParamsBuffer = this.device.createBuffer({
      size: CLUSTER_PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Per cluster: u32 count + MAX_PER_CLUSTER u32 light indices.
    this.clusterLightsBuffer = this.device.createBuffer({
      size: CLUSTER_COUNT * (MAX_PER_CLUSTER + 1) * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.clusterBindGroup = this.device.createBindGroup({
      layout: this.clusterLayout,
      entries: [
        { binding: 0, resource: { buffer: this.clusterParamsBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: { buffer: this.clusterLightsBuffer } },
      ],
    });
    this.buildFrameBindGroup(); // binding 14 now points at the real list
  }

  /** Encode the clustered-lighting light-assignment compute pass. */
  private prepareClusters(encoder: GPUCommandEncoder, camera: Camera): void {
    if (!this.clusteredLighting) return;
    this.ensureClusterResources();

    const params = new Float32Array(CLUSTER_PARAMS_SIZE / 4);
    params.set(camera.matrixWorldInverse.elements, 0);
    params.set(camera.projectionMatrixInverse.elements, 16);
    params[32] = this.frameData[35]; // packed light count (set by uploadFrame)
    const cam = camera as unknown as { near?: number; far?: number };
    params[33] = Math.max(cam.near ?? 0.1, 0.001);
    params[34] = cam.far ?? 2000;
    this.device.queue.writeBuffer(this.clusterParamsBuffer!, 0, params);

    const pass = encoder.beginComputePass({ label: 'clusters' });
    pass.setPipeline(this.clusterPipeline!);
    pass.setBindGroup(0, this.clusterBindGroup!);
    pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64));
    pass.end();
  }

  private ensureGpuCullResources(minSlots: number): void {
    if (!this.gpuCullPipeline) {
      const module = this.device.createShaderModule({ code: CULL_SHADER, label: 'cull' });
      this.gpuCullLayout = this.device.createBindGroupLayout({
        label: 'cull',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      });
      this.gpuCullPipeline = this.device.createComputePipeline({
        label: 'frustum-cull',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.gpuCullLayout] }),
        compute: { module, entryPoint: 'cs_cull' },
      });
      this.gpuCullParamsBuffer = this.device.createBuffer({
        size: CULL_PARAMS_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    if (this.gpuCullCapacity >= minSlots) return;
    const cap = Math.max(minSlots, this.gpuCullCapacity * 2 || 256);
    this.gpuSphereBuffer?.destroy();
    this.gpuSphereBuffer = this.device.createBuffer({
      size: cap * 16, // vec4 per slot
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.gpuIndirectBuffer?.destroy();
    this.gpuIndirectBuffer = this.device.createBuffer({
      size: cap * INDIRECT_STRIDE,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.gpuCullCapacity = cap;
    this.gpuCullBindGroup = null;
  }

  private prepareGpuCull(encoder: GPUCommandEncoder): void {
    if (!this.gpuCulling || this.opaque.length === 0) return;

    // Pre-upload sphere data and draw params for all opaque meshes.
    for (const mesh of this.opaque) {
      if (mesh instanceof InstancedMesh) continue;
      const slot = this.getMeshSlot(mesh); // uploads model matrix, assigns slot
      this.ensureGpuCullResources(slot + 1);

      // World-space bounding sphere.
      const geo = mesh.geometry;
      if (!geo.boundingSphere) geo.computeBoundingSphere();
      const bs = geo.boundingSphere;
      const sphere = new Float32Array(4);
      if (bs && !bs.isEmpty()) {
        this._worldSphere.copy(bs).applyMatrix4(mesh.matrixWorld);
        sphere[0] = this._worldSphere.center.x;
        sphere[1] = this._worldSphere.center.y;
        sphere[2] = this._worldSphere.center.z;
        sphere[3] = this._worldSphere.radius;
      } else {
        sphere[3] = -1; // no valid sphere → always cull
      }
      this.device.queue.writeBuffer(this.gpuSphereBuffer!, slot * 16, sphere);

      // Draw params (instanceCount=1; compute shader may set to 0).
      const gpuGeo = this.geometries.get(mesh.geometry);
      if (gpuGeo.index) {
        const cmd = new Uint32Array([gpuGeo.drawCount, 1, 0, 0, 0]);
        this.device.queue.writeBuffer(this.gpuIndirectBuffer!, slot * INDIRECT_STRIDE, cmd);
      }
    }

    // Upload frustum planes + draw count.
    const paramsBuf = new ArrayBuffer(CULL_PARAMS_SIZE);
    const pf = new Float32Array(paramsBuf);
    const pu = new Uint32Array(paramsBuf);
    for (let i = 0; i < 6; i++) {
      const pl = this._frustum.planes[i];
      pf[i * 4 + 0] = pl.normal.x;
      pf[i * 4 + 1] = pl.normal.y;
      pf[i * 4 + 2] = pl.normal.z;
      pf[i * 4 + 3] = pl.constant;
    }
    pu[24] = this.nextMeshSlot; // drawCount at byte 96
    this.device.queue.writeBuffer(this.gpuCullParamsBuffer!, 0, paramsBuf);

    // Rebuild bind group if buffers were (re)created.
    if (!this.gpuCullBindGroup) {
      this.gpuCullBindGroup = this.device.createBindGroup({
        layout: this.gpuCullLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.gpuCullParamsBuffer! } },
          { binding: 1, resource: { buffer: this.gpuSphereBuffer! } },
          { binding: 2, resource: { buffer: this.gpuIndirectBuffer! } },
        ],
      });
    }

    const pass = encoder.beginComputePass({ label: 'frustum-cull' });
    pass.setPipeline(this.gpuCullPipeline!);
    pass.setBindGroup(0, this.gpuCullBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nextMeshSlot / 64));
    pass.end();
  }

  private clearColor(scene: Scene): GPUColor {
    const bg = scene.background;
    if (!bg) return { r: 0.05, g: 0.05, b: 0.06, a: 1 };
    // background is linear; encode to sRGB-ish for the non-srgb target
    const enc = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
    return { r: enc(bg.r), g: enc(bg.g), b: enc(bg.b), a: 1 };
  }

  private collect(scene: Scene): void {
    this.opaque.length = 0;
    this.transparent.length = 0;
    this.lights.length = 0;
    this.culledCount = 0;

    scene.traverseVisible((object: Object3D) => {
      if (object instanceof Light) {
        this.lights.push(object);
      } else if (object instanceof Mesh) {
        if (this.frustumCulling && object.frustumCulled && this.isCulled(object)) {
          this.culledCount++;
          return;
        }
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        const transparent = materials.some((m) => m.transparent);
        if (transparent) this.transparent.push(object);
        else this.opaque.push(object);
      }
    });
  }

  /** True if the mesh's world-space bounding sphere lies outside the frustum. */
  private isCulled(mesh: Mesh): boolean {
    const geometry = mesh.geometry;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere || sphere.isEmpty()) return false; // can't bound → never cull
    this._worldSphere.copy(sphere).applyMatrix4(mesh.matrixWorld);
    return !this._frustum.intersectsSphere(this._worldSphere);
  }

  private uploadFrame(scene: Scene, camera: Camera): void {
    const f = this.frameData;
    f.set(camera.matrixWorldInverse.elements, 0); // view  (0..15)
    f.set(camera.projectionMatrix.elements, 16); // proj  (16..31)

    // TAA sub-pixel jitter: shift the projection so geometry moves by
    // (_jitterX, _jitterY) in NDC.
    if (this._jitterX !== 0 || this._jitterY !== 0) {
      if (f[16 + 15] === 1) {
        // Orthographic (w_clip = 1): jitter via the translation column.
        f[16 + 12] += this._jitterX;
        f[16 + 13] += this._jitterY;
      } else {
        // Perspective (w_clip = -z_view): NDC shift is -e[8] / -e[9].
        f[16 + 8] -= this._jitterX;
        f[16 + 9] -= this._jitterY;
      }
    }

    camera.getWorldPosition(this._camPos);
    f[32] = this._camPos.x;
    f[33] = this._camPos.y;
    f[34] = this._camPos.z;

    // ambient accumulation
    let ar = scene.ambientColor.r * scene.ambientIntensity;
    let ag = scene.ambientColor.g * scene.ambientIntensity;
    let ab = scene.ambientColor.b * scene.ambientIntensity;

    let lightCount = 0;
    const ld = this.lightData;
    for (const light of this.lights) {
      if (light instanceof AmbientLight) {
        ar += light.color.r * light.intensity;
        ag += light.color.g * light.intensity;
        ab += light.color.b * light.intensity;
        continue;
      }
      if (lightCount >= MAX_LIGHTS) break;
      const base = lightCount * (LIGHT_STRIDE / 4);
      light.getWorldPosition(this._meshPos);

      if (light instanceof DirectionalLight) {
        light.target.updateWorldMatrix(true, false);
        const tx = light.target.matrixWorld.elements[12];
        const ty = light.target.matrixWorld.elements[13];
        const tz = light.target.matrixWorld.elements[14];
        const dir = new Vector3(tx - this._meshPos.x, ty - this._meshPos.y, tz - this._meshPos.z).normalize();
        ld[base + 3] = 0; // kind = directional
        ld[base + 4] = dir.x;
        ld[base + 5] = dir.y;
        ld[base + 6] = dir.z;
        ld[base + 7] = 0; // range unused
        ld[base + 8] = light.color.r * light.intensity;
        ld[base + 9] = light.color.g * light.intensity;
        ld[base + 10] = light.color.b * light.intensity;
        ld[base + 11] = 1; // decay unused
      } else if (light instanceof PointLight) {
        ld[base + 0] = this._meshPos.x;
        ld[base + 1] = this._meshPos.y;
        ld[base + 2] = this._meshPos.z;
        ld[base + 3] = 1; // kind = point
        ld[base + 7] = light.distance; // range
        ld[base + 8] = light.color.r * light.intensity;
        ld[base + 9] = light.color.g * light.intensity;
        ld[base + 10] = light.color.b * light.intensity;
        ld[base + 11] = light.decay;
        ld[base + 12] = 1; ld[base + 13] = 1;
        // spotParams.z: first of 6 consecutive cube-face shadow tiles (−1 = none).
        const pc = this.spotShadowCasters.find(c => c.packedIndex === lightCount);
        ld[base + 14] = pc ? pc.tileIndex : -1;
        ld[base + 15] = 0;
      } else if (light instanceof SpotLight) {
        light.target.updateWorldMatrix(true, false);
        const tx = light.target.matrixWorld.elements[12];
        const ty = light.target.matrixWorld.elements[13];
        const tz = light.target.matrixWorld.elements[14];
        const dir = new Vector3(tx - this._meshPos.x, ty - this._meshPos.y, tz - this._meshPos.z).normalize();
        ld[base + 0] = this._meshPos.x;
        ld[base + 1] = this._meshPos.y;
        ld[base + 2] = this._meshPos.z;
        ld[base + 3] = 2; // kind = spot
        ld[base + 4] = dir.x;
        ld[base + 5] = dir.y;
        ld[base + 6] = dir.z;
        ld[base + 7] = light.distance;
        ld[base + 8] = light.color.r * light.intensity;
        ld[base + 9] = light.color.g * light.intensity;
        ld[base + 10] = light.color.b * light.intensity;
        ld[base + 11] = light.decay;
        ld[base + 12] = Math.cos(light.angle * (1 - light.penumbra)); // cosInner
        ld[base + 13] = Math.cos(light.angle);                         // cosOuter
        // Look up this light's shadow tile index (−1 if none).
        const sc = this.spotShadowCasters.find(c => c.packedIndex === lightCount);
        ld[base + 14] = sc ? sc.tileIndex : -1;
        ld[base + 15] = 0;
      }
      lightCount++;
    }

    f[35] = lightCount; // cameraPos.w
    f[36] = ar;
    f[37] = ag;
    f[38] = ab;
    f[39] = this.exposure; // ambient.w

    // lightViewProj (40..55) + shadowParams (56..59)
    if (this.shadowCasterIndex >= 0) {
      f.set(this._lightViewProj.elements, 40);
      f[56] = 1; // enabled
      f[57] = this.shadowMapSize;
      f[58] = this.shadowNormalBias;
      f[59] = this.shadowCasterIndex;
    } else {
      f[56] = 0;
    }

    // envParams (60..63); w = linear output for the post pipeline.
    f[60] = this.envEnabled ? 1 : 0;
    f[61] = this.envIntensity;
    f[62] = this.envMaxMip;
    // Bit 0: linear output; bit 1: IBL active; bit 2: screen-space refraction available.
    f[63] = (this.postProcessing ? 1 : 0) | (this.iblActive ? 2 : 0) | (this.postProcessing ? 4 : 0);

    // clusterParams (64..67) + clusterDims (68..71)
    const cam = camera as unknown as { near?: number; far?: number };
    f[64] = this.clusteredLighting ? 1 : 0;
    f[65] = Math.max(cam.near ?? 0.1, 0.001);
    f[66] = cam.far ?? 2000;
    f[67] = scene.backgroundBlur; // skybox blur (0 = sharp, 1 = max mip)
    f[68] = this.canvas.width / CLUSTER_X;  // tile size in pixels
    f[69] = this.canvas.height / CLUSTER_Y;
    f[70] = (performance.now() - this.clockStart) / 1000; // elapsed seconds (shader elapsedTime())
    f[71] = 0;

    this.device.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    if (lightCount > 0) {
      this.device.queue.writeBuffer(this.lightBuffer, 0, this.lightData, 0, lightCount * (LIGHT_STRIDE / 4));
    }
  }

  exposure = 1.0;

  private drawMesh(pass: DrawEncoder, mesh: Mesh, oit = false): void {
    const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (material instanceof LineBasicMaterial) {
      if (!oit) this.drawLine(pass, mesh, material); // lines aren't part of the OIT pass
      return;
    }
    const isShader = material instanceof ShaderMaterial;
    if (!isShader && !(material instanceof StandardMaterial)) return;
    const geometry = this.geometries.get(mesh.geometry);

    const instanced = mesh instanceof InstancedMesh;
    const skinned = !instanced && mesh instanceof SkinnedMesh && geometry.joints !== null && geometry.weights !== null;
    const morphed =
      !instanced && !skinned &&
      mesh.morphTargetInfluences.length > 0 &&
      !!mesh.geometry.morphAttributes.position?.length;
    const variant = instanced ? 'instanced' : skinned ? 'skinned' : morphed ? 'morph' : 'static';

    const oitSampleCount = oit ? this.sampleCount : 1;
    if (isShader) {
      const sm = material as ShaderMaterial;
      const res = this.getShaderMaterialResources(sm);
      const cacheKey = `sm:${sm.id}:v${sm.version}:${res.shapeKey}`;
      pass.setPipeline(this.pipelines.getCustom(
        cacheKey, variant, sm, this.sceneTargetFormat, oit, oitSampleCount,
        () => buildSurfaceShader(variant, res.layout.wgsl, sm.surfaceCode),
        sm.name || sm.type,
      ));
      pass.setBindGroup(2, res.bindGroup);
    } else {
      const std = material as StandardMaterial;
      pass.setPipeline(oit
        ? this.pipelines.getOIT(std, variant, oitSampleCount)
        : this.pipelines.get(std, variant, this.sceneTargetFormat));
      pass.setBindGroup(2, this.getMaterialResources(std).bindGroup);
    }

    // Group 1: model uniform (static/skinned/morph) or instance storage (instanced)
    if (instanced) {
      pass.setBindGroup(1, this.getInstancedResources(mesh as InstancedMesh).bindGroup);
    } else {
      pass.setBindGroup(1, this.modelPoolBindGroup!, [this.getMeshSlot(mesh) * 256]);
    }

    pass.setVertexBuffer(0, geometry.position);
    pass.setVertexBuffer(1, geometry.normal);
    pass.setVertexBuffer(2, geometry.uv);
    pass.setVertexBuffer(3, geometry.tangent);

    if (skinned) {
      pass.setBindGroup(3, this.getSkinnedResources(mesh as SkinnedMesh).bindGroup);
      pass.setVertexBuffer(4, geometry.joints!);
      pass.setVertexBuffer(5, geometry.weights!);
      pass.setVertexBuffer(6, geometry.color); // color follows the skinning streams
    } else {
      pass.setVertexBuffer(4, geometry.color);
      if (morphed) pass.setBindGroup(3, this.getMorphResources(mesh).bindGroup);
    }

    const instanceCount = instanced ? (mesh as InstancedMesh).count : 1;
    if (geometry.index) {
      pass.setIndexBuffer(geometry.index, geometry.indexFormat);
      // For GPU-driven culling, use an indirect draw so instanceCount can be zeroed by the
      // compute shader.  Only works with a render pass encoder (not render bundles).
      if (this.gpuCulling && !instanced && !oit && pass instanceof GPURenderPassEncoder) {
        const slot = this.meshSlots.get(mesh) ?? 0;
        pass.drawIndexedIndirect(this.gpuIndirectBuffer!, slot * INDIRECT_STRIDE);
      } else {
        pass.drawIndexed(geometry.drawCount, instanceCount);
      }
    } else {
      pass.draw(geometry.drawCount, instanceCount);
    }
  }

  private drawLine(pass: DrawEncoder, mesh: Mesh, material: LineBasicMaterial): void {
    // The color stream is always present (white default), so lines need no setup.
    const geometry = this.geometries.get(mesh.geometry);

    pass.setPipeline(this.pipelines.getLine(material, this.sceneTargetFormat));
    pass.setBindGroup(1, this.modelPoolBindGroup!, [this.getMeshSlot(mesh) * 256]);
    pass.setBindGroup(2, this.getLineResources(material).bindGroup);
    pass.setVertexBuffer(0, geometry.position);
    pass.setVertexBuffer(1, geometry.color);

    if (geometry.index) {
      pass.setIndexBuffer(geometry.index, geometry.indexFormat);
      pass.drawIndexed(geometry.drawCount);
    } else {
      pass.draw(geometry.drawCount);
    }
  }

  /** Get (or re-record) the opaque render bundle when its draw set changes. */
  private getOpaqueBundle(): GPURenderBundle {
    const colorFormat = this.postProcessing ? 'rgba16float' : this.format;
    let key = `${colorFormat}|${this.frameBindGroupVersion}|`;
    for (const mesh of this.opaque) {
      const m = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      key += `${mesh.id},${mesh.geometry.id},${(m as Material).id}`;
      // ShaderMaterial recompiles invalidate the recorded pipeline.
      if (m instanceof ShaderMaterial) key += `v${m.version}`;
      key += ';';
    }
    if (this.opaqueBundle && key === this.bundleKey) return this.opaqueBundle;

    const encoder = this.device.createRenderBundleEncoder({
      colorFormats: [colorFormat],
      depthStencilFormat: DEPTH_FORMAT,
      sampleCount: this.sampleCount,
    });
    encoder.setBindGroup(0, this.frameBindGroup); // bundles don't inherit pass state
    for (const mesh of this.opaque) this.drawMesh(encoder, mesh);
    this.opaqueBundle = encoder.finish();
    this.bundleKey = key;
    return this.opaqueBundle;
  }

  /** Refresh per-object GPU buffers each frame so a replayed bundle stays correct. */
  private refreshOpaqueUniforms(): void {
    for (const mesh of this.opaque) {
      const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (material instanceof LineBasicMaterial) {
        this.getMeshSlot(mesh);
        this.getLineResources(material);
        continue;
      }
      const isShader = material instanceof ShaderMaterial;
      if (!isShader && !(material instanceof StandardMaterial)) continue;
      if (mesh instanceof InstancedMesh) this.getInstancedResources(mesh);
      else this.getMeshSlot(mesh);
      if (isShader) this.getShaderMaterialResources(material as ShaderMaterial);
      else this.getMaterialResources(material as StandardMaterial);
      const geometry = this.geometries.get(mesh.geometry);
      if (!(mesh instanceof InstancedMesh) && mesh instanceof SkinnedMesh &&
          geometry.joints !== null && geometry.weights !== null) {
        this.getSkinnedResources(mesh);
      } else if (!(mesh instanceof InstancedMesh) && mesh.morphTargetInfluences.length > 0 &&
                 mesh.geometry.morphAttributes.position?.length) {
        this.getMorphResources(mesh);
      }
    }
  }

  private getLineResources(material: LineBasicMaterial): { buffer: GPUBuffer; bindGroup: GPUBindGroup } {
    let res = this.lineResources.get(material);
    if (!res) {
      const buffer = this.device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.lineMaterialLayout,
        entries: [{ binding: 0, resource: { buffer } }],
      });
      res = { buffer, bindGroup };
      this.lineResources.set(material, res);
    }
    const data = new Float32Array(8);
    data[0] = material.color.r;
    data[1] = material.color.g;
    data[2] = material.color.b;
    data[3] = material.opacity;
    data[4] = material.vertexColors ? 1 : 0;
    this.device.queue.writeBuffer(res.buffer, 0, data);
    return res;
  }

  private getInstancedResources(mesh: InstancedMesh): InstancedResources {
    let res = this.instancedResources.get(mesh);
    if (!res || res.count !== mesh.count) {
      const buffer = this.device.createBuffer({
        size: Math.max(mesh.count, 1) * 64,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.instanceLayout,
        entries: [{ binding: 0, resource: { buffer } }],
      });
      res = { buffer, bindGroup, count: mesh.count, version: -1 };
      this.instancedResources.set(mesh, res);
    }
    if (res.version !== mesh.version) {
      this.device.queue.writeBuffer(res.buffer, 0, mesh.instanceMatrix);
      res.version = mesh.version;
    }
    return res;
  }

  private getSkinnedResources(mesh: SkinnedMesh): SkinnedResources {
    const skeleton = mesh.skeleton;
    let res = this.skinnedResources.get(mesh);
    if (!res || res.jointCount !== skeleton.jointCount) {
      const boneBuffer = this.device.createBuffer({
        size: Math.max(skeleton.jointCount, 1) * 64,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.bonesLayout,
        entries: [{ binding: 0, resource: { buffer: boneBuffer } }],
      });
      res = { boneBuffer, bindGroup, jointCount: skeleton.jointCount };
      this.skinnedResources.set(mesh, res);
    }
    // Joints are already updated by scene.updateMatrixWorld(); refresh bones.
    skeleton.update();
    this.device.queue.writeBuffer(res.boneBuffer, 0, skeleton.boneMatrices);
    return res;
  }

  private getMorphResources(mesh: Mesh): MorphResources {
    const geometry = mesh.geometry;
    const positions = geometry.morphAttributes.position!;
    const count = positions.length;

    let res = this.morphResources.get(mesh);
    if (!res || res.count !== count) {
      const vertexCount = geometry.attributes.position.count;
      const normals = geometry.morphAttributes.normal;
      const hasNormals = normals && normals.length === count ? 1 : 0;

      // Pack per-target deltas contiguously: [target][vertex][xyz].
      const packed = (attrs: typeof positions | undefined): Float32Array<ArrayBuffer> => {
        const out = new Float32Array(count * vertexCount * 3);
        if (attrs) {
          for (let t = 0; t < count; t++) out.set(attrs[t].array, t * vertexCount * 3);
        }
        return out;
      };
      const posData = packed(positions);
      const nrmData = hasNormals ? packed(normals) : new Float32Array(1);

      const storage = (data: Float32Array<ArrayBuffer>): GPUBuffer => {
        const buffer = this.device.createBuffer({
          size: Math.max(data.byteLength, 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
      };
      const posBuffer = storage(posData);
      const nrmBuffer = storage(nrmData);

      const infoBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(infoBuffer, 0, new Uint32Array([count, vertexCount, hasNormals, 0]));

      const weightBuffer = this.device.createBuffer({
        size: Math.max(count * 4, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.morphLayout,
        entries: [
          { binding: 0, resource: { buffer: infoBuffer } },
          { binding: 1, resource: { buffer: posBuffer } },
          { binding: 2, resource: { buffer: nrmBuffer } },
          { binding: 3, resource: { buffer: weightBuffer } },
        ],
      });
      res = { infoBuffer, weightBuffer, bindGroup, count };
      this.morphResources.set(mesh, res);
    }

    // Weights change every frame (animation); deltas/info are static.
    this.device.queue.writeBuffer(res.weightBuffer, 0, new Float32Array(mesh.morphTargetInfluences));
    return res;
  }

  private ensureModelPool(minSlots: number): void {
    if (this.modelPoolCapacity >= minSlots) return;
    const capacity = Math.max(minSlots, this.modelPoolCapacity * 2, 64);
    this.modelPoolBuffer?.destroy();
    this.modelPoolBuffer = this.device.createBuffer({
      size: capacity * 256, // 256-byte aligned slots, MODEL_SIZE (128) per slot
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.modelPoolBindGroup = this.device.createBindGroup({
      layout: this.pipelines.modelLayout,
      entries: [{ binding: 0, resource: { buffer: this.modelPoolBuffer, size: MODEL_SIZE } }],
    });
    this.modelPoolCapacity = capacity;
    // Invalidate any recorded render bundle (pool bind group changed).
    this.opaqueBundle = null;
    this.bundleKey = '';
  }

  /** Assign a stable pool slot and upload the current model/normal matrix. Returns slot index. */
  private getMeshSlot(mesh: Mesh): number {
    let slot = this.meshSlots.get(mesh);
    if (slot === undefined) {
      slot = this.nextMeshSlot++;
      this.meshSlots.set(mesh, slot);
    }
    this.ensureModelPool(slot + 1);

    const data = new Float32Array(MODEL_SIZE / 4);
    data.set(mesh.matrixWorld.elements, 0);
    this._normalMatrix.getNormalMatrix(mesh.matrixWorld);
    const nm = this._normalMatrix.elements;
    data[16] = nm[0]; data[17] = nm[1]; data[18] = nm[2]; data[19] = 0;
    data[20] = nm[3]; data[21] = nm[4]; data[22] = nm[5]; data[23] = 0;
    data[24] = nm[6]; data[25] = nm[7]; data[26] = nm[8]; data[27] = 0;
    data[28] = 0; data[29] = 0; data[30] = 0; data[31] = 1;
    this.device.queue.writeBuffer(this.modelPoolBuffer!, slot * 256, data);
    return slot;
  }

  private getMaterialResources(material: StandardMaterial): MaterialResources {
    const signature = this.textureSignature(material);
    let res = this.materialResources.get(material);

    if (!res || res.textureSignature !== signature) {
      const uniformBuffer =
        res?.uniformBuffer ??
        this.device.createBuffer({
          size: MATERIAL_SIZE,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

      const t = (map: StandardMaterial['map'], def: GPUTextureView) =>
        map ? this.textures.get(map).view : def;
      const s = (map: StandardMaterial['map']) =>
        map ? this.textures.get(map).sampler : this.textures.defaultSampler;

      const white = this.textures.defaultWhiteView;
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.materialLayout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: t(material.map, white) },
          { binding: 2, resource: s(material.map) },
          { binding: 3, resource: t(material.normalMap, this.textures.defaultNormalView) },
          { binding: 4, resource: s(material.normalMap) },
          { binding: 5, resource: t(material.metalnessRoughnessMap, white) },
          { binding: 6, resource: s(material.metalnessRoughnessMap) },
          { binding: 7, resource: t(material.emissiveMap, white) },
          { binding: 8, resource: s(material.emissiveMap) },
          { binding: 9, resource: t(material.occlusionMap, white) },
          { binding: 10, resource: s(material.occlusionMap) },
          { binding: 11, resource: t(material.clearcoatMap, white) },
          { binding: 12, resource: s(material.clearcoatMap) },
          { binding: 13, resource: t(material.clearcoatRoughnessMap, white) },
          { binding: 14, resource: s(material.clearcoatRoughnessMap) },
        ],
      });
      res = { uniformBuffer, bindGroup, textureSignature: signature };
      this.materialResources.set(material, res);
    }

    // update uniforms each frame (cheap)
    const data = new Float32Array(MATERIAL_SIZE / 4);
    data[0] = material.color.r;
    data[1] = material.color.g;
    data[2] = material.color.b;
    data[3] = material.opacity;
    data[4] = material.emissive.r;
    data[5] = material.emissive.g;
    data[6] = material.emissive.b;
    data[7] = material.emissiveIntensity;
    data[8] = material.metalness;
    data[9] = material.roughness;
    data[10] = material.normalScale;
    data[11] = material.occlusionStrength;
    data[12] = material.alphaTest;
    let flags = 0;
    if (material.normalMap) flags |= 2;
    data[13] = flags;
    data[14] = material.clearcoat; // misc.z
    data[15] = material.clearcoatRoughness; // misc.w
    data[16] = material.specularColor.r; // specular.rgb = specular color factor
    data[17] = material.specularColor.g;
    data[18] = material.specularColor.b;
    data[19] = material.specularIntensity; // specular.w = specular factor
    data[20] = material.ior; // extra.x
    data[24] = material.sheenColor.r; // sheen.rgb = sheen color factor
    data[25] = material.sheenColor.g;
    data[26] = material.sheenColor.b;
    data[27] = material.sheenRoughness; // sheen.w = sheen roughness
    data[28] = material.transmission; // transmission.x
    data[29] = material.thickness; // transmission.y
    data[30] = material.attenuationDistance; // transmission.z
    data[32] = material.attenuationColor.r; // attenuation.rgb
    data[33] = material.attenuationColor.g;
    data[34] = material.attenuationColor.b;
    this.device.queue.writeBuffer(res.uniformBuffer, 0, data);
    return res;
  }

  private textureSignature(m: StandardMaterial): string {
    const id = (t: StandardMaterial['map']) => (t ? `${t.id}:${t.version}` : '_');
    return [m.map, m.normalMap, m.metalnessRoughnessMap, m.emissiveMap, m.occlusionMap,
            m.clearcoatMap, m.clearcoatRoughnessMap]
      .map(id)
      .join('|');
  }

  /**
   * Uniform buffer + bind group for a ShaderMaterial. Recreated when the
   * uniforms object's shape (names/types) changes — which also changes the
   * pipeline cache key, recompiling the shader to match. Values are packed
   * and uploaded once per frame.
   */
  private getShaderMaterialResources(material: ShaderMaterial): ShaderMaterialResources {
    const layout = computeUniformLayout(material.uniforms);
    const shapeKey = layout.fields.map((f) => `${f.name}:${f.kind}`).join(',');
    let res = this.shaderMaterialResources.get(material);
    if (!res || res.shapeKey !== shapeKey) {
      res?.buffer.destroy();
      const buffer = this.device.createBuffer({
        size: layout.size,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.customUniformLayout,
        entries: [{ binding: 0, resource: { buffer } }],
      });
      res = { buffer, bindGroup, data: new Float32Array(layout.size / 4), layout, shapeKey, uploadedFrame: -1 };
      this.shaderMaterialResources.set(material, res);
    }
    if (res.uploadedFrame !== this.frameNumber) {
      packUniforms(material.uniforms, res.layout, res.data);
      this.device.queue.writeBuffer(res.buffer, 0, res.data);
      res.uploadedFrame = this.frameNumber;
    }
    return res;
  }

  // ---------------------------------------------------------------------------
  // GPU id-buffer picking
  // ---------------------------------------------------------------------------

  private ensureIdResources(meshCount: number): void {
    if (!this.idLayout) {
      this.idLayout = this.device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 4 },
        }],
      });
    }

    const needed = Math.max(meshCount, 1) * 256;
    if (this.idBufferCapacity < needed) {
      this.idUniformBuffer?.destroy();
      this.idUniformBuffer = this.device.createBuffer({
        size: needed,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.idBindGroup = this.device.createBindGroup({
        layout: this.idLayout,
        entries: [{ binding: 0, resource: { buffer: this.idUniformBuffer, size: 4 } }],
      });
      this.idBufferCapacity = needed;
    }

    if (!this.idPipeline) {
      const module = this.device.createShaderModule({ code: ID_SHADER });
      const layout = this.device.createPipelineLayout({
        bindGroupLayouts: [this.pipelines.frameLayout, this.pipelines.modelLayout, this.idLayout],
      });
      this.idPipeline = this.device.createRenderPipeline({
        layout,
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }],
        },
        fragment: {
          module,
          entryPoint: 'fs_main',
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      });
    }
  }

  private ensureIdTextures(): void {
    const w = this.canvas.width;
    const h = this.canvas.height;
    if (this.idTexW !== w || this.idTexH !== h) {
      this.idColorTexture?.destroy();
      this.idDepthTexture?.destroy();
      this.idColorTexture = this.device.createTexture({
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.idDepthTexture = this.device.createTexture({
        size: [w, h],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.idTexW = w;
      this.idTexH = h;
    }
    if (!this.idReadBuffer) {
      this.idReadBuffer = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /**
   * Render an offscreen id-buffer pass and read back the pixel at (cssX, cssY)
   * to identify which mesh lies under the cursor. Returns null for background.
   * Skinned and morph deformations are not reflected; positions use the mesh
   * world matrix only. InstancedMesh is excluded (returns null for instances).
   */
  async pickAt(cssX: number, cssY: number, scene: Scene, camera: Camera): Promise<Mesh | null> {
    scene.updateMatrixWorld();
    camera.updateMatrixWorld();
    this.collect(scene);
    this.uploadFrame(scene, camera);

    const meshList: Mesh[] = [];
    scene.traverseVisible((obj: Object3D) => {
      if (obj instanceof Mesh && !(obj instanceof InstancedMesh)) {
        const mat = Array.isArray(obj.material) ? obj.material[0] : obj.material;
        if (mat instanceof StandardMaterial || mat instanceof ShaderMaterial) meshList.push(obj);
      }
    });

    this.ensureIdResources(meshList.length);
    this.ensureIdTextures();

    // Pre-upload all mesh ids. Each id occupies 256 bytes (uniform alignment).
    // id=0 means background; meshes get ids 1..N at 256-byte strides.
    const idData = new Uint32Array(meshList.length * 64);
    for (let i = 0; i < meshList.length; i++) idData[i * 64] = i + 1;
    this.device.queue.writeBuffer(this.idUniformBuffer!, 0, idData);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.idColorTexture!.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view: this.idDepthTexture!.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this.idPipeline!);
    pass.setBindGroup(0, this.frameBindGroup);
    for (let i = 0; i < meshList.length; i++) this.drawMeshId(pass, meshList[i], i);
    pass.end();

    const px = Math.max(0, Math.min(this.idTexW - 1, Math.floor(cssX * this.pixelRatio)));
    const py = Math.max(0, Math.min(this.idTexH - 1, Math.floor(cssY * this.pixelRatio)));
    encoder.copyTextureToBuffer(
      { texture: this.idColorTexture!, origin: { x: px, y: py } },
      { buffer: this.idReadBuffer!, bytesPerRow: 256 },
      { width: 1, height: 1 },
    );

    this.device.queue.submit([encoder.finish()]);
    await this.idReadBuffer!.mapAsync(GPUMapMode.READ);
    const pixel = new Uint8Array(this.idReadBuffer!.getMappedRange(0, 4));
    const id = pixel[0] | (pixel[1] << 8) | (pixel[2] << 16);
    this.idReadBuffer!.unmap();

    return id > 0 && id <= meshList.length ? meshList[id - 1] : null;
  }

  private drawMeshId(pass: GPURenderPassEncoder, mesh: Mesh, index: number): void {
    const geometry = this.geometries.get(mesh.geometry);
    pass.setBindGroup(1, this.modelPoolBindGroup!, [this.getMeshSlot(mesh) * 256]);
    pass.setBindGroup(2, this.idBindGroup!, [index * 256]);
    pass.setVertexBuffer(0, geometry.position);
    if (geometry.index) {
      pass.setIndexBuffer(geometry.index, geometry.indexFormat);
      pass.drawIndexed(geometry.drawCount);
    } else {
      pass.draw(geometry.drawCount);
    }
  }
}
