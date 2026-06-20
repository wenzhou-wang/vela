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
import { ParticleSystem } from '../core/ParticleSystem';
import { Sprite } from '../core/Sprite';
import { LOD } from '../core/LOD';
import { TextMesh } from '../core/TextMesh';
import { RenderTarget } from '../core/RenderTarget';
import { ReflectionProbe } from '../core/ReflectionProbe';
import { IrradianceProbeGrid } from '../core/IrradianceProbeGrid';
import { PerspectiveCamera } from '../core/PerspectiveCamera';
import { SDFFontAtlas, SDF_BASE_FONT } from '../textures/SDFFontAtlas';
import type { Material } from '../materials/Material';
import { StandardMaterial } from '../materials/StandardMaterial';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { ShaderMaterial, computeUniformLayout, packUniforms } from '../materials/ShaderMaterial';
import type { UniformLayout } from '../materials/ShaderMaterial';
import { Texture } from '../textures/Texture';
import { ShaderPass } from './ShaderPass';
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
import { PostProcessing, type ToneMapping } from './PostProcessing';
import type { ColorLUT } from '../textures/ColorLUT';
import { IBLPrefilter, IBL_MIP_LEVELS } from './IBLPrefilter';
import { CULL_SHADER } from './shaders/cull.wgsl';
import { HIZ_COPY_SHADER, HIZ_DOWN_SHADER } from './shaders/hiz.wgsl';
import { CLUSTER_SHADER } from './shaders/clusters.wgsl';
import { SKYGEN_SHADER } from './shaders/skygen.wgsl';
import { REFLECTION_PROBE_SHADER } from './shaders/reflectionProbe.wgsl';
import { IRRADIANCE_PROBE_SHADER } from './shaders/irradianceProbe.wgsl';
import { VOLUMETRIC_FOG_SHADER } from './shaders/volumetricFog.wgsl';
import { PARTICLE_SIM_SHADER } from './shaders/particles.wgsl';
import { buildSurfaceShader } from './shaders/surface.wgsl';
import { diagnoseScene, type Diagnostic } from './diagnose';

/** Either encoder accepts the same draw commands (pass or render bundle). */
type DrawEncoder = GPURenderPassEncoder | GPURenderBundleEncoder;

const MAX_LIGHTS = 256;
const UNIT_Y = new Vector3(0, 1, 0);
const UNIT_Z = new Vector3(0, 0, 1);
const FRAME_SIZE = 320; // bytes (view/proj/lightViewProj mat4s + cameraPos..fogParams vec4s)
const MODEL_SIZE = 144; // model mat4 (64) + normalMat mat4 (64) + params vec4 (16)
const MATERIAL_SIZE = 144;
const LIGHT_STRIDE = 64; // bytes per light (positionKind + directionRange + colorDecay + spotParams)

// GPU-driven culling: indirect draw + compute-shader sphere cull.
// 6×vec4 planes (96) + prevViewProj mat4 (64) + drawCount/occlusion/hizMips/pad
// (16) + hizSize vec2 + pad (16) = 192.
const CULL_PARAMS_SIZE = 192;
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
const PROBE_FACE_UPS = [
  new Vector3(0, -1, 0), new Vector3(0, -1, 0),
  new Vector3(0, 0, 1), new Vector3(0, 0, -1),
  new Vector3(0, -1, 0), new Vector3(0, -1, 0),
];

// TAA: 8-sample Halton(2,3) sub-pixel jitter pattern.
const TAA_SAMPLES = 8;

// Procedural sky: equirect texture size (workgroup 8x8 must divide both).
const SKY_TEX_WIDTH = 256;
const SKY_TEX_HEIGHT = 128;
const MAX_REFLECTION_PROBES = 4;

// Sprite instance stride: posPad + sizeOffset + color + uvRect (4 × vec4).
const SPRITE_STRIDE_F = 16;

// Render targets store tonemapped sRGB-encoded bytes; sampling goes through an
// -srgb view so materials read linear values back.
const RT_FORMAT: GPUTextureFormat = 'rgba8unorm';

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

/** Practical CSM split scheme: lambda=0 uniform, lambda=1 logarithmic. */
export function computeCascadeSplits(
  near: number,
  far: number,
  count: number,
  lambda = 0.6,
  out = new Float32Array(4),
): Float32Array {
  const n = Math.min(4, Math.max(1, Math.floor(count)));
  const clampedNear = Math.max(near, 0.001);
  const blend = Math.min(1, Math.max(0, lambda));
  for (let i = 0; i < n; i++) {
    const p = (i + 1) / n;
    const logarithmic = clampedNear * Math.pow(far / clampedNear, p);
    const uniform = clampedNear + (far - clampedNear) * p;
    out[i] = logarithmic * blend + uniform * (1 - blend);
  }
  for (let i = n; i < 4; i++) out[i] = far;
  return out;
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
  buffer: GPUBuffer | null; // null when the material has no scalar uniforms
  bindGroup: GPUBindGroup;
  data: Float32Array<ArrayBuffer>;
  layout: UniformLayout;
  shapeKey: string;       // uniform names+kinds; a change rebuilds buffer + shader
  uploadedFrame: number;  // last frameNumber the uniform values were written
  textureSig: string;     // texture ids+versions; a change rebuilds the bind group
}

interface SpriteBatch {
  screen: boolean;
  sdf: boolean;
  textureSig: string;                // `${texture.id}:${texture.version}` or 'white'
  data: Float32Array<ArrayBuffer>;   // SPRITE_STRIDE_F floats per instance
  capacity: number;
  count: number;
  buffer: GPUBuffer;
  params: GPUBuffer;
  bindGroup: GPUBindGroup;
}

interface RenderTargetResources {
  color: GPUTexture;                     // rgba8unorm color (readable via readPixels)
  renderView: GPUTextureView;            // attachment / resolve target view
  msaaView: GPUTextureView | null;       // MSAA color when sampleCount > 1
  depthView: GPUTextureView;
}

interface ReflectionProbeResources {
  equirect: GPUTexture;
  equirectView: GPUTextureView;
  ibl: IBLPrefilter;
  lastCaptureFrame: number;
  slot: number;
}

/** An actionable performance suggestion (same shape as a Diagnostic). */
export interface PerfSuggestion {
  /** Stable identifier (e.g. 'instancing-opportunity') for programmatic handling. */
  code: string;
  /** What was observed. */
  message: string;
  /** The recommended change. */
  fix: string;
}

/** Last-frame statistics from `renderer.report()` — JSON an agent can reason over. */
export interface RenderReport {
  /** Monotonic frame counter. */
  frame: number;
  /** Scene-pass draw calls (excludes shadow passes; see shadowDraws). */
  drawCalls: number;
  /** Triangles submitted (instancing included). */
  triangles: number;
  meshes: { opaque: number; transparent: number; culled: number };
  lights: number;
  particles: { systems: number; poolCapacity: number };
  sprites: { batches: number; instances: number };
  /** Extra draws spent re-rendering casters into shadow maps. */
  shadowDraws: number;
  /** Estimated GPU texture memory in bytes. */
  textureMemoryBytes: number;
  flags: { postProcessing: boolean; clusteredLighting: boolean; ssr: boolean; msaa: number; shadows: boolean; volumetricFog: boolean };
  /** Actionable performance suggestions for the last frame (may be empty). */
  suggestions: PerfSuggestion[];
}

/** RGBA8 pixel readback result (rows top-to-bottom). */
export interface PixelData {
  data: Uint8ClampedArray<ArrayBuffer>;
  width: number;
  height: number;
}

export interface IrradianceBakeOptions {
  /** Cube-face resolution used before SH projection. Default 64. */
  resolution?: number;
  near?: number;
  far?: number;
}

interface ParticleResources {
  capacity: number;
  stateBuffer: GPUBuffer;   // Particle[] pool (zeroed = all dead)
  simParams: GPUBuffer;
  simBindGroup: GPUBindGroup;
  drawParams: GPUBuffer;
  drawBindGroup: GPUBindGroup;
  cursor: number;           // ring-buffer emission cursor
  carry: number;            // fractional particles owed from previous frames
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
  /** Number of directional shadow cascades. Values above 1 require a perspective camera. */
  shadowCascades = 1;
  /** Blend between logarithmic (1) and uniform (0) cascade split placement. */
  shadowCascadeLambda = 0.6;
  /** Fraction of each cascade depth range used to blend into the next cascade. */
  shadowCascadeBlend = 0.1;
  private shadowTexture!: GPUTexture;
  private shadowView!: GPUTextureView;
  private shadowSampler!: GPUSampler;
  private cascadeBuffer!: GPUBuffer;
  private cascadeLightBuffers: GPUBuffer[] = [];
  private cascadeLightBindGroups: GPUBindGroup[] = [];
  private cascadeViewProjs = [new Matrix4(), new Matrix4(), new Matrix4(), new Matrix4()];
  private cascadeCount = 1;
  private cascadeSplits = new Float32Array(4);
  private cascadeCorners = Array.from({ length: 8 }, () => new Vector3());
  /** Integrate scene fog and clustered lights into a low-resolution 3-D froxel volume. */
  volumetricFog = false;
  private volumetricFogTexture!: GPUTexture;
  private volumetricFogView!: GPUTextureView;
  private volumetricFogSampler!: GPUSampler;
  private volumetricFogPipeline: GPUComputePipeline | null = null;
  private volumetricFogLayout: GPUBindGroupLayout | null = null;
  private volumetricFogBindGroup: GPUBindGroup | null = null;
  private volumetricFogClusterBuffer: GPUBuffer | null = null;
  private volumetricFogInvVP!: GPUBuffer;
  private _inverseViewProjection = new Matrix4();
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
  private reflectionProbeResources = new WeakMap<ReflectionProbe, ReflectionProbeResources>();
  private reflectionProbeTexture!: GPUTexture;
  private reflectionProbeView!: GPUTextureView;
  private reflectionProbeSampler!: GPUSampler;
  private reflectionProbeBuffer!: GPUBuffer;
  private reflectionProbeData = new Float32Array(24); // 4 positions/radii + 4 intensities + count
  private reflectionProbePipeline: GPUComputePipeline | null = null;
  private reflectionProbeLayout: GPUBindGroupLayout | null = null;
  private capturingReflectionProbe = false;
  private irradianceGridBuffer!: GPUBuffer;
  private irradianceGridParams!: GPUBuffer;
  private irradianceGridCapacity = 0;
  private irradianceGridKey = '';
  private irradianceProbePipeline: GPUComputePipeline | null = null;
  private irradianceProbeLayout: GPUBindGroupLayout | null = null;

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
   * Output transform for the post pipeline: `'aces'` filmic (default), `'agx'`
   * (AgX operator — gentler highlight desaturation), or `'none'` for sRGB-only
   * output (flat/stylized looks). Applies when `postProcessing` is on.
   */
  toneMapping: ToneMapping = 'aces';
  /**
   * Optional 3-D color-grading LUT applied as the final post pass (after tonemap,
   * in display space). Requires `postProcessing`. Load one with
   * `ColorLUT.parseCube(text)`.
   */
  colorLUT: ColorLUT | null = null;
  /**
   * Custom fullscreen post effects, run in order in HDR linear space before
   * tonemap (requires `postProcessing`). Push `ShaderPass` instances to add
   * effects; reorder or splice to change the chain.
   */
  passes: ShaderPass[] = [];
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
  /** Screen-space reflections. Requires `postProcessing = true` and `sampleCount = 1`. */
  ssr = false;
  /** Strength of the reflected HDR contribution. */
  ssrIntensity = 0.5;
  /** Maximum view-space ray length in world units. */
  ssrMaxDistance = 50;
  /** View-space hit tolerance in world units. */
  ssrThickness = 0.2;
  /**
   * Temporal anti-aliasing: a sub-pixel Halton jitter is baked into the
   * projection matrix and a resolve pass blends the reprojected history
   * (camera-motion only). Requires `postProcessing = true` and `sampleCount = 1`;
   * otherwise silently disabled.
   */
  taa = false;
  /** TAA blend factor: weight of the current frame (lower = smoother, more ghosting). */
  taaBlend = 0.1;
  /**
   * Deterministic rendering: identical scenes produce identical pixels.
   * Time stops following the wall clock — drive it via `renderer.time`
   * (e.g. `renderer.time += 1 / 60` per frame); particle simulation and
   * shader `elapsedTime()` use it, and the TAA jitter sequence restarts from
   * a fixed point when this is enabled. The foundation for golden-image tests.
   */
  deterministic = false;
  /** Scene time in seconds; only advances when you set it (`deterministic = true`). */
  time = 0;
  private lastDeterministicTime = 0;
  private prevDeterministic = false;
  private _elapsed = 0; // seconds fed to shaders/particles this render
  private taaActive = false;     // was TAA running last frame (history validity)
  private taaFrameIndex = 0;     // Halton sequence cursor
  private _jitterX = 0;          // NDC jitter baked into this frame's projection
  private _jitterY = 0;
  private _prevViewProj = new Float32Array(16);
  private _invViewProj = new Matrix4();
  private _jitteredProj = new Matrix4();
  private _invJitteredProj = new Matrix4();
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
   * Hi-Z occlusion culling (opt-in, builds on `gpuCulling`). Each frame a
   * max-depth pyramid is built from the scene depth; the next frame's GPU cull
   * tests each bounding sphere's screen footprint against it and skips meshes
   * fully behind closer geometry. Requires `gpuCulling = true` and
   * `sampleCount = 1` (the depth must be sampleable); silently inactive
   * otherwise. Conservative max-reduction, but uses the previous frame's depth,
   * so objects revealed by fast camera motion can pop in one frame late.
   */
  occlusionCulling = false;
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
  // Hi-Z occlusion: max-depth pyramid built each frame, used the next.
  private hizTexture: GPUTexture | null = null;
  private hizMipViews: GPUTextureView[] = []; // per-mip render-attachment views
  private hizSampleView: GPUTextureView | null = null; // full-chain sampling view
  private hizSampler: GPUSampler | null = null;
  private hizDummyView: GPUTextureView | null = null; // 1×1, bound when no pyramid yet
  private hizCopyPipeline: GPURenderPipeline | null = null;
  private hizDownPipeline: GPURenderPipeline | null = null;
  private hizCopyLayout: GPUBindGroupLayout | null = null;
  private hizDownLayout: GPUBindGroupLayout | null = null;
  private hizMipCount = 0;
  private hizValid = false; // a pyramid from a prior frame is available
  private _boundHizView: GPUTextureView | null = null;
  private _prevHizViewProj = new Float32Array(16); // viewProj that built the bound hi-Z
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
  private shellSlots = new WeakMap<Mesh, number>();
  private nextMeshSlot = 0;

  // Lifecycle.
  private disposed = false;
  private deviceLost = false;
  /** Called when the GPU device is lost unexpectedly; call `restoreContext()` to recover. */
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;

  private materialResources = new WeakMap<Material, MaterialResources>();
  private shaderMaterialResources = new WeakMap<ShaderMaterial, ShaderMaterialResources>();
  // Skybox: frame uniform + raw env view (rebuilt when the environment changes).
  private skyBindGroup: GPUBindGroup | null = null;
  private skyBindGroupKey = '';
  // Procedural sky generation (lazy; reused across regenerations).
  private skyGenPipeline: GPUComputePipeline | null = null;
  private skyGenParamsBuffer: GPUBuffer | null = null;
  private skyGenBindGroup: GPUBindGroup | null = null;
  private skyGenView: GPUTextureView | null = null;
  private skyGenSampler: GPUSampler | null = null;
  // GPU particles.
  private particleSystems: ParticleSystem[] = [];
  private particleResources = new WeakMap<ParticleSystem, ParticleResources>();
  private particleSimPipeline: GPUComputePipeline | null = null;
  private particleSimLayout: GPUBindGroupLayout | null = null;
  private lastRenderTime = 0;
  // Sprites: one instanced batch per (texture, screen-space, sdf) combination.
  private sprites: Sprite[] = [];
  private spriteBatches = new Map<string, SpriteBatch>();
  // SDF text: glyphs flow through the sprite batcher, one atlas per font.
  private texts: TextMesh[] = [];
  private fontAtlases = new Map<string, SDFFontAtlas>();
  // Scene color format for this frame's pipelines (HDR under post-processing).
  private sceneTargetFormat: GPUTextureFormat = 'bgra8unorm';
  private frameNumber = 0;
  private readonly clockStart = performance.now();
  // Per-render state: output dimensions + whether the post chain runs.
  private _renderWidth = 1;
  private _renderHeight = 1;
  private _usePost = false;
  private _sceneInputs = false;
  private renderTargets = new WeakMap<RenderTarget, RenderTargetResources>();
  // readPixels() waiters fulfilled with the next presented canvas frame.
  private pendingCapture: Array<(p: PixelData) => void> = [];
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
  /** Whether any collected mesh has a shell (skips the shell loop otherwise). */
  private hasShells = false;
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
    await this.acquireDevice();
    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.setupDeviceState();
    this.setSize(this.canvas.clientWidth || 800, this.canvas.clientHeight || 600);
  }

  /** Request an adapter + device and install the device-lost handler. */
  private async acquireDevice(): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No suitable GPUAdapter found.');
    this.device = await adapter.requestDevice();
    this.deviceLost = false;
    this.device.lost.then((info) => {
      // 'destroyed' = we called dispose()/destroy() ourselves; not an error.
      if (this.disposed || info.reason === 'destroyed') return;
      this.deviceLost = true;
      console.error(
        `[vela] WebGPU device lost: ${info.message}\n` +
        'Rendering is paused. Call await renderer.restoreContext() to rebuild GPU ' +
        'resources (geometry/material/texture caches repopulate lazily), or set ' +
        'renderer.onDeviceLost to handle it.',
      );
      this.onDeviceLost?.(info);
    });
  }

  /**
   * (Re)configure the canvas, (re)create every device-owned object, and reset
   * all lazy caches. The single source of truth for "what lives on the GPU" —
   * shared by `init()` and `restoreContext()`. Caches keyed by surviving
   * CPU-side objects (geometry/material/texture data) repopulate on next draw.
   */
  private setupDeviceState(): void {
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
      // COPY_SRC lets readPixels()/screenshot() capture the presented frame.
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    // Device-owned managers (fresh instances drop all their internal caches).
    this.geometries = new GeometryBuffers(this.device);
    this.textures = new TextureManager(this.device);
    this.pipelines = new PipelineCache(
      this.device, this.format, this.sampleCount,
      PBR_SHADER, PBR_SKINNED_SHADER, PBR_INSTANCED_SHADER, PBR_MORPH_SHADER, LINE_SHADER, SHADOW_SHADER,
    );
    this.post = new PostProcessing(this.device, this.format, this.sampleCount);
    this.ibl = new IBLPrefilter(this.device);

    // Reset renderer-held lazy caches so they rebuild against the new device.
    this.materialResources = new WeakMap();
    this.shaderMaterialResources = new WeakMap();
    this.skinnedResources = new WeakMap();
    this.instancedResources = new WeakMap();
    this.morphResources = new WeakMap();
    this.lineResources = new WeakMap();
    this.particleResources = new WeakMap();
    this.renderTargets = new WeakMap();
    this.reflectionProbeResources = new WeakMap();
    this.meshSlots = new WeakMap();
    this.shellSlots = new WeakMap();
    this.nextMeshSlot = 0;
    this.spriteBatches.clear();
    this.fontAtlases.clear();

    // Null lazily-created device singletons + reset their size/capacity guards.
    this.clusterPipeline = null; this.clusterParamsBuffer = null;
    this.clusterLightsBuffer = null; this.clusterDummyBuffer = null; this.clusterBindGroup = null;
    this.gpuCullPipeline = null; this.gpuCullLayout = null; this.gpuCullParamsBuffer = null;
    this.gpuSphereBuffer = null; this.gpuIndirectBuffer = null; this.gpuCullBindGroup = null;
    this.gpuCullCapacity = 0;
    this.hizTexture = null; this.hizMipViews = []; this.hizSampleView = null;
    this.hizSampler = null; this.hizDummyView = null;
    this.hizCopyPipeline = null; this.hizDownPipeline = null;
    this.hizCopyLayout = null; this.hizDownLayout = null;
    this.hizMipCount = 0; this.hizValid = false; this._boundHizView = null;
    this.modelPoolBuffer = null; this.modelPoolBindGroup = null; this.modelPoolCapacity = 0;
    this.skyBindGroup = null; this.skyBindGroupKey = '';
    this.skyGenPipeline = null; this.skyGenParamsBuffer = null; this.skyGenBindGroup = null;
    this.skyGenView = null; this.skyGenSampler = null;
    this.reflectionProbePipeline = null; this.reflectionProbeLayout = null;
    this.irradianceProbePipeline = null; this.irradianceProbeLayout = null;
    this.irradianceGridCapacity = 0; this.irradianceGridKey = '';
    this.volumetricFogPipeline = null; this.volumetricFogLayout = null;
    this.volumetricFogBindGroup = null; this.volumetricFogClusterBuffer = null;
    this.particleSimPipeline = null; this.particleSimLayout = null;
    this.idPipeline = null; this.idLayout = null; this.idUniformBuffer = null;
    this.idBindGroup = null; this.idBufferCapacity = 0;
    this.idColorTexture = null; this.idDepthTexture = null; this.idReadBuffer = null;
    this.idTexW = 0; this.idTexH = 0;
    this.shadowMapAllocated = 0; this.spotAtlasAllocated = 0;
    this.shadowSampler = null as unknown as GPUSampler;
    this.cascadeBuffer = null as unknown as GPUBuffer;
    this.cascadeLightBuffers = [];
    this.cascadeLightBindGroups = [];
    this.spotAtlasSampler = null;
    this.spotShadowTilesBuffer = null;
    this.spotLightBuffers = [];
    this.spotLightBindGroups = [];
    this.opaqueBundle = null; this.bundleKey = '';
    this.envKey = ''; this.iblEnvKey = ''; this.iblActive = false;
    // Null the frame buffer so createShadowResources() doesn't rebuild the frame
    // bind group before createFrameResources() recreates its dependencies.
    this.frameBuffer = null as unknown as GPUBuffer;
    this.lightBuffer = null as unknown as GPUBuffer;

    this.createShadowResources();
    this.envView = this.textures.defaultWhiteView;
    this.envSampler = this.textures.defaultSampler;
    // Compute the BRDF LUT once.
    const brdfEncoder = this.device.createCommandEncoder();
    this.ibl.computeBRDFLUT(brdfEncoder);
    this.device.queue.submit([brdfEncoder.finish()]);
    this.createReflectionProbeResources();
    this.createIrradianceGridResources();
    this.createVolumetricFogResources();
    this.createFrameResources();
  }

  /**
   * Rebuild after a lost device (or to recover from a GPU reset). Requests a
   * fresh device, reconfigures the canvas, and recreates all GPU resources;
   * meshes/materials/textures redraw from their surviving CPU-side data. Safe
   * to call from `onDeviceLost`. No-op if the renderer was disposed.
   */
  async restoreContext(): Promise<void> {
    if (this.disposed) return;
    await this.acquireDevice();
    this.setupDeviceState();
    const w = this.width, h = this.height;
    this.width = 0; this.height = 0; // force size-dependent rebuild
    this.setSize(w, h);
  }

  /**
   * Release GPU resources and destroy the device. The renderer is unusable
   * afterward; create a new one to render again.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.spriteBatches.clear();
    this.fontAtlases.clear();
    this.device?.destroy();
  }

  /** True once `dispose()` has run. */
  get isDisposed(): boolean { return this.disposed; }
  /** True while the device is lost and not yet restored. */
  get isDeviceLost(): boolean { return this.deviceLost; }

  /**
   * Debug snapshot of tracked GPU resource usage — for spotting leaks in
   * long-running sessions. Counts reflect live cache entries.
   */
  resources(): { textureMemoryBytes: number; spriteBatches: number; fontAtlases: number; modelPoolSlots: number } {
    return {
      textureMemoryBytes: this.textures?.totalBytes ?? 0,
      spriteBatches: this.spriteBatches.size,
      fontAtlases: this.fontAtlases.size,
      modelPoolSlots: this.nextMeshSlot,
    };
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
        { binding: 8, resource: this.ibl.brdfLUTView },
        { binding: 9, resource: iblSampler },
        // bindings 10-12: spot-light shadow atlas (dummy when not allocated)
        { binding: 10, resource: { buffer: this.spotShadowTilesBuffer! } },
        { binding: 11, resource: this.spotAtlasView! },
        { binding: 12, resource: this.spotAtlasSampler! },
        // binding 13: screen-space refraction capture (post.sceneCaptureView is always valid)
        { binding: 13, resource: this.post.sceneCaptureView },
        // binding 14: clustered light lists (dummy until clustered lighting runs)
        { binding: 14, resource: { buffer: this.clusterLightsBuffer ?? this.clusterDummyBuffer! } },
        // bindings 15-17: local reflection-probe array, sampler, and positions.
        { binding: 15, resource: this.reflectionProbeView },
        { binding: 16, resource: this.reflectionProbeSampler },
        { binding: 17, resource: { buffer: this.reflectionProbeBuffer } },
        { binding: 18, resource: { buffer: this.irradianceGridBuffer } },
        { binding: 19, resource: { buffer: this.irradianceGridParams } },
        { binding: 20, resource: { buffer: this.cascadeBuffer } },
        { binding: 21, resource: this.volumetricFogView },
        { binding: 22, resource: this.volumetricFogSampler },
      ],
    });
  }

  private createReflectionProbeResources(): void {
    this.reflectionProbeTexture = this.device.createTexture({
      label: 'reflection-probes',
      size: { width: 256, height: 128, depthOrArrayLayers: MAX_REFLECTION_PROBES },
      mipLevelCount: IBL_MIP_LEVELS,
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.reflectionProbeView = this.reflectionProbeTexture.createView({ dimension: '2d-array' });
    this.reflectionProbeSampler = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge',
    });
    this.reflectionProbeBuffer = this.device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.reflectionProbeBuffer, 0, this.reflectionProbeData);
  }

  private createIrradianceGridResources(): void {
    this.irradianceGridCapacity = 144;
    this.irradianceGridBuffer = this.device.createBuffer({
      label: 'irradiance-grid-sh', size: this.irradianceGridCapacity,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.irradianceGridParams = this.device.createBuffer({
      label: 'irradiance-grid-params', size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private createVolumetricFogResources(): void {
    this.volumetricFogTexture = this.device.createTexture({
      label: 'volumetric-fog',
      size: { width: CLUSTER_X, height: CLUSTER_Y, depthOrArrayLayers: CLUSTER_Z },
      dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.volumetricFogView = this.volumetricFogTexture.createView({ dimension: '3d' });
    this.volumetricFogSampler = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge',
    });
    this.volumetricFogInvVP = this.device.createBuffer({
      size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
      this.volumetricFogBindGroup = null;
      this.shadowMapAllocated = size;
      if (this.frameBuffer) this.buildFrameBindGroup(); // shadowView changed → rebind
    }
    if (!this.shadowSampler) {
      this.shadowSampler = this.device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    }
    if (this.cascadeLightBuffers.length === 0) {
      for (let i = 0; i < 4; i++) {
        const buffer = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.cascadeLightBuffers.push(buffer);
        this.cascadeLightBindGroups.push(this.device.createBindGroup({
          layout: this.pipelines.shadowLightLayout,
          entries: [{ binding: 0, resource: { buffer } }],
        }));
      }
    }
    if (!this.cascadeBuffer) {
      this.cascadeBuffer = this.device.createBuffer({
        size: 288, // 4 mat4s + split vec4 + params vec4
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

  render(scene: Scene, camera: Camera, target?: RenderTarget): void {
    // Skip rendering while the device is gone; restoreContext()/dispose() govern recovery.
    if (this.disposed || this.deviceLost) return;
    scene.updateMatrixWorld();
    camera.updateMatrixWorld();
    if (!this.capturingReflectionProbe && !target) {
      this.refreshReflectionProbes(scene);
      // Probe renders update shared per-render state; the main render rebuilds it below.
      scene.updateMatrixWorld();
      camera.updateMatrixWorld();
    }
    this.frameNumber++;
    // Frame timing: wall clock normally; renderer.time when deterministic.
    let dt: number;
    if (this.deterministic) {
      if (!this.prevDeterministic) {
        this.lastDeterministicTime = this.time;
        this.taaFrameIndex = 0;               // restart the jitter sequence
        this.post?.invalidateTAAHistory();
      }
      dt = Math.max(this.time - this.lastDeterministicTime, 0);
      this.lastDeterministicTime = this.time;
      this._elapsed = this.time;
    } else {
      const nowMs = performance.now();
      dt = this.lastRenderTime > 0 ? Math.min((nowMs - this.lastRenderTime) / 1000, 0.1) : 0;
      this.lastRenderTime = nowMs;
      this._elapsed = (nowMs - this.clockStart) / 1000;
    }
    this.prevDeterministic = this.deterministic;
    // Render-target passes use the direct pipeline; the post chain (and its
    // screen-sized resources) belongs to the default canvas target.
    const rt = target ? this.getRenderTargetResources(target) : null;
    const usePost = this.postProcessing && !rt;
    this._usePost = usePost;
    const useSSR = this.ssr && usePost && this.sampleCount === 1 && !!this.depthSampleView;
    this._sceneInputs = usePost && (useSSR || this.passes.some((p) => p.enabled && p.inputs.length > 0));
    this._renderWidth = target ? target.width : this.canvas.width;
    this._renderHeight = target ? target.height : this.canvas.height;
    // The scene pass renders into the HDR target when post-processing is on,
    // so material pipelines must target that format.
    this.sceneTargetFormat = usePost ? 'rgba16float' : rt ? RT_FORMAT : this.format;

    // Build the view frustum for culling (projection * view).
    this._viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._viewProjection);

    // TAA: pick this frame's sub-pixel jitter (baked into the uploaded projection).
    const useTAA = this.taa && usePost && this.sampleCount === 1 && !!this.depthSampleView;
    if (useTAA) {
      const i = (this.taaFrameIndex++ % TAA_SAMPLES) + 1;
      this._jitterX = (halton(i, 2) * 2 - 1) / this._renderWidth;
      this._jitterY = (halton(i, 3) * 2 - 1) / this._renderHeight;
      if (!this.taaActive) this.post?.invalidateTAAHistory(); // just (re)enabled
    } else {
      this._jitterX = 0;
      this._jitterY = 0;
    }
    if (!rt) this.taaActive = useTAA;

    // Pick LOD levels by camera distance before collecting (toggles child visibility).
    camera.getWorldPosition(this._camPos);
    scene.traverse((o) => {
      if (o instanceof LOD && o.autoUpdate) o.update(this._camPos);
    });

    this.collect(scene);
    this.prepareShadow(camera);
    this.prepareReflectionProbeBindings(scene);
    this.prepareIrradianceGrid(scene);
    this.uploadFrame(scene, camera);
    this.prepareSprites();

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
    this.prepareVolumetricFog(encoder, scene, camera);
    // GPU particles: emit + integrate the pools.
    this.prepareParticles(encoder, dt);

    // Don't acquire the swap-chain texture for offscreen renders.
    const swapTexture = rt ? null : this.context.getCurrentTexture();
    const swapView = swapTexture ? swapTexture.createView() : null;
    const outView = rt ? rt.renderView : swapView!;
    const clear = this.clearColor(scene);

    let colorAttachment: GPURenderPassColorAttachment;
    if (usePost) {
      this.post.ensureSize(this.canvas.width, this.canvas.height);
      colorAttachment = this.post.sceneColorAttachment(clear);
    } else if (this.sampleCount > 1) {
      colorAttachment = {
        view: rt ? rt.msaaView! : this.msaaTexture!.createView(),
        resolveTarget: outView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: clear,
      };
    } else {
      colorAttachment = { view: outView, loadOp: 'clear', storeOp: 'store', clearValue: clear };
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: this._sceneInputs
        ? [colorAttachment, this.post.sceneNormalDepthAttachment()]
        : [colorAttachment],
      depthStencilAttachment: {
        view: rt ? rt.depthView : this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setBindGroup(0, this.frameBindGroup);

    if (this.renderBundles && this.opaque.length && !rt) {
      this.refreshOpaqueUniforms(); // keep per-object buffers fresh for the replay
      pass.executeBundles([this.getOpaqueBundle()]);
    } else {
      for (const mesh of this.opaque) this.drawMesh(pass, mesh);
    }

    // Inverted-hull shells (outlines/fur/highlights): extruded back-face draws
    // after the opaques, so the depth test keeps only the silhouette ring.
    if (this.hasShells) {
      for (const mesh of this.opaque) if (mesh.shell) this.drawShell(pass, mesh);
      for (const mesh of this.transparent) if (mesh.shell) this.drawShell(pass, mesh);
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
      pass.setPipeline(this.pipelines.getSky(this.sceneTargetFormat, this._sceneInputs));
      pass.setBindGroup(0, this.skyBindGroup);
      pass.draw(3);
      pass.setBindGroup(0, this.frameBindGroup); // restore for transparent draws
    }

    const useOIT = this.oit && usePost && this.transparent.length > 0;
    if (!useOIT) {
      for (const mesh of this.transparent) this.drawMesh(pass, mesh);
    }

    // Particles: blended billboards, depth-tested against the opaque scene.
    if (this.particleSystems.length > 0) this.drawParticles(pass);
    // Sprites & text: world-space labels, then HUD overlays on top.
    if (this.spriteBatches.size > 0) this.drawSprites(pass);

    pass.end();

    // Capture the opaque HDR scene for screen-space refraction before transparent draws.
    if (usePost && this.transparent.length > 0) {
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

    const useSSAO = this.ssao && usePost && this.sampleCount === 1 && !!this.depthSampleView;
    // SSR needs the current frame's full hi-Z chain. Occlusion culling consumes
    // the previous contents earlier in this command buffer, before this rebuild.
    if (!rt && (useSSR || (this.gpuCulling && this.occlusionCulling)) &&
        this.sampleCount === 1 && this.depthSampleView) {
      this.buildHiZ(encoder);
    }
    // Resolve the HDR target through the post chain into the swap chain.
    if (usePost) {
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
      let postInput = this.post.hdrTargetView;
      if (useSSR) {
        this._jitteredProj.fromArray(this.frameData, 16);
        this._invJitteredProj.copy(this._jitteredProj).invert();
        postInput = this.post.runSSR(
          encoder,
          postInput,
          this.post.sceneNormalDepthView,
          this.hizSampleView!,
          {
            proj: new Float32Array(this._jitteredProj.elements),
            invProj: new Float32Array(this._invJitteredProj.elements),
            view: new Float32Array(camera.matrixWorldInverse.elements),
          },
          this.ssrMaxDistance,
          this.ssrThickness,
          this.ssrIntensity,
          this.hizMipCount,
        );
      }
      // TAA resolve runs after SSR so its history stabilizes ray-march shimmer.
      if (useTAA) {
        this._invViewProj.copy(this._viewProjection).invert();
        postInput = this.post.runTAA(
          encoder,
          postInput,
          this.depthSampleView!,
          this._prevViewProj,
          new Float32Array(this._invViewProj.elements),
          this._jitterX,
          this._jitterY,
          this.taaBlend,
        );
      }
      // Custom ShaderPasses run in HDR linear space before tonemap.
      if (this.passes.length > 0) {
        const cam = camera as unknown as { near?: number; far?: number };
        postInput = this.post.runShaderPasses(
          encoder,
          this.passes,
          postInput,
          this.depthSampleView ?? this.post.dummyDepth,
          this._sceneInputs ? this.post.sceneNormalDepthView : this.post.dummyNormalDepth,
          this.textures,
          this._elapsed,
          {
            proj: new Float32Array(camera.projectionMatrix.elements),
            invProj: new Float32Array(camera.projectionMatrixInverse.elements),
            view: new Float32Array(camera.matrixWorldInverse.elements),
            near: cam.near ?? 0.1,
            far: cam.far ?? 2000,
          },
        );
      }
      this.post.run(encoder, swapView!, {
        fxaa: this.fxaa,
        bloom: this.bloom,
        bloomThreshold: this.bloomThreshold,
        bloomIntensity: this.bloomIntensity,
        ssao: useSSAO,
        ssaoStrength: this.ssaoStrength,
        toneMapping: this.toneMapping,
        colorLUT: this.colorLUT,
      }, postInput);
    }
    // Save this frame's unjittered view-projection for next frame's TAA
    // reprojection (canvas renders only — offscreen cameras must not pollute it).
    if (!rt) this._prevViewProj.set(this._viewProjection.elements);

    // Pending readPixels(): copy the final frame into a mappable buffer before
    // the swap texture is presented.
    let captureBuffer: GPUBuffer | null = null;
    if (!rt && this.pendingCapture.length > 0) {
      const w = this.canvas.width;
      const h = this.canvas.height;
      const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
      captureBuffer = this.device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyTextureToBuffer(
        { texture: swapTexture! },
        { buffer: captureBuffer, bytesPerRow },
        [w, h],
      );
    }

    this.device.queue.submit([encoder.finish()]);

    if (captureBuffer) {
      const resolvers = this.pendingCapture;
      this.pendingCapture = [];
      this.resolveCapture(captureBuffer, this.canvas.width, this.canvas.height, this.format === 'bgra8unorm', resolvers);
    }
  }

  /** Map the capture buffer, strip row padding (+ BGRA swizzle), resolve waiters. */
  private resolveCapture(
    buffer: GPUBuffer,
    width: number,
    height: number,
    bgra: boolean,
    resolvers: Array<(p: PixelData) => void>,
  ): void {
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    buffer.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Uint8Array(buffer.getMappedRange());
      const data = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++) {
        const row = y * bytesPerRow;
        const out = y * width * 4;
        if (bgra) {
          for (let x = 0; x < width; x++) {
            data[out + x * 4] = src[row + x * 4 + 2];
            data[out + x * 4 + 1] = src[row + x * 4 + 1];
            data[out + x * 4 + 2] = src[row + x * 4];
            data[out + x * 4 + 3] = src[row + x * 4 + 3];
          }
        } else {
          data.set(src.subarray(row, row + width * 4), out);
        }
      }
      buffer.unmap();
      buffer.destroy();
      for (const resolve of resolvers) resolve({ data, width, height });
    });
  }

  /**
   * Last-frame statistics as plain JSON: draw calls, triangles, culling
   * results, memory estimates, and active feature flags. Reflects the most
   * recent `render()` call.
   */
  report(): RenderReport {
    let triangles = 0;
    const countMesh = (mesh: Mesh): void => {
      const geo = mesh.geometry;
      const verts = geo.index ? geo.index.count : (geo.attributes.position?.count ?? 0);
      const instances = mesh instanceof InstancedMesh ? mesh.count : 1;
      triangles += Math.floor(verts / 3) * instances;
    };
    for (const m of this.opaque) countMesh(m);
    for (const m of this.transparent) countMesh(m);

    let spriteInstances = 0;
    for (const b of this.spriteBatches.values()) spriteInstances += b.count;
    let particleCapacity = 0;
    for (const sys of this.particleSystems) particleCapacity += sys.options.capacity ?? 1000;

    const shadowTiles = this.spotShadowCasters.length + (this.shadowCasterIndex >= 0 ? this.cascadeCount : 0);
    const drawCalls = this.opaque.length + this.transparent.length +
      this.particleSystems.length + this.spriteBatches.size;
    return {
      frame: this.frameNumber,
      drawCalls,
      triangles,
      meshes: {
        opaque: this.opaque.length,
        transparent: this.transparent.length,
        culled: this.culledCount,
      },
      lights: this.lights.length,
      particles: { systems: this.particleSystems.length, poolCapacity: particleCapacity },
      sprites: { batches: this.spriteBatches.size, instances: spriteInstances },
      shadowDraws: shadowTiles * this.opaque.length,
      textureMemoryBytes: this.textures?.totalBytes ?? 0,
      flags: {
        postProcessing: this.postProcessing,
        clusteredLighting: this.clusteredLighting,
        ssr: this.ssr && this.postProcessing && this.sampleCount === 1,
        msaa: this.sampleCount,
        shadows: this.shadows,
        volumetricFog: this.volumetricFog,
      },
      suggestions: this.perfSuggestions(drawCalls),
    };
  }

  /** Heuristic performance advice for the last frame (the report's `suggestions`). */
  private perfSuggestions(drawCalls: number): PerfSuggestion[] {
    const out: PerfSuggestion[] = [];

    // Many non-instanced meshes sharing one geometry+material → InstancedMesh.
    const groups = new Map<string, number>();
    for (const mesh of this.opaque) {
      if (mesh instanceof InstancedMesh) continue;
      const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (!mat) continue;
      const key = `${mesh.geometry.id}|${mat.id}`;
      groups.set(key, (groups.get(key) ?? 0) + 1);
    }
    let bestCount = 0;
    for (const count of groups.values()) {
      if (count > bestCount) bestCount = count;
    }
    if (bestCount >= 16) {
      out.push({
        code: 'instancing-opportunity',
        message: `${bestCount} meshes share one geometry+material but draw separately (${drawCalls} total draw calls).`,
        fix: 'Replace them with a single InstancedMesh (one draw for the whole batch).',
      });
    }

    // High draw-call count without render bundles.
    if (drawCalls > 500 && !this.renderBundles) {
      out.push({
        code: 'consider-render-bundles',
        message: `${drawCalls} draw calls per frame with renderBundles off — CPU encoding may dominate.`,
        fix: 'Set renderer.renderBundles = true to record and replay the opaque pass.',
      });
    }

    // Many ranged lights without clustered lighting.
    let rangedLights = 0;
    for (const l of this.lights) {
      const range = (l as { distance?: number }).distance;
      if (range !== undefined && range > 0) rangedLights++;
    }
    if (rangedLights >= 16 && !this.clusteredLighting) {
      out.push({
        code: 'consider-clustered-lighting',
        message: `${rangedLights} ranged point/spot lights with clusteredLighting off — every fragment loops all lights.`,
        fix: 'Set renderer.clusteredLighting = true (perspective cameras).',
      });
    }

    // Post effects requested but post pipeline off (silently inactive).
    if (!this.postProcessing && (this.bloom || this.ssao || this.ssr || this.taa || this.oit || this.colorLUT || this.passes.length > 0)) {
      out.push({
        code: 'post-effect-inactive',
        message: 'A post effect (bloom/ssao/ssr/taa/oit/colorLUT/ShaderPass) is enabled but renderer.postProcessing is off — it does nothing.',
        fix: 'Set renderer.postProcessing = true, or disable the effect to avoid confusion.',
      });
    }

    // Heavy MSAA while features that need sampleCount 1 are requested.
    if (this.sampleCount > 1 && (this.ssao || this.ssr || this.taa)) {
      out.push({
        code: 'msaa-blocks-effect',
        message: `sampleCount is ${this.sampleCount}× but SSAO/SSR/TAA require sampleCount 1 — those effects are inactive.`,
        fix: 'Recreate the renderer with { sampleCount: 1 } to use SSAO/SSR/TAA.',
      });
    }

    return out;
  }

  /**
   * Explain why a scene "renders but looks wrong" — especially black screens.
   * Returns structured findings, each with a stable `code`, a message naming
   * the offending object, and the one-line `fix`. Empty array = nothing
   * suspicious. Cheap enough to call when stuck; not meant for every frame.
   */
  diagnose(scene: Scene, camera: Camera): Diagnostic[] {
    return diagnoseScene(scene, camera, {
      postProcessing: this.postProcessing,
      sampleCount: this.sampleCount,
      oit: this.oit,
      ssao: this.ssao,
      ssr: this.ssr,
      taa: this.taa,
      shadows: this.shadows,
      shadowCascades: this.shadowCascades,
      volumetricFog: this.volumetricFog,
      canvasWidth: this.canvas.width,
      canvasHeight: this.canvas.height,
    });
  }

  /**
   * Read back rendered pixels as RGBA8 (rows top-to-bottom).
   *
   * - `readPixels()` captures the **next canvas frame** — call `render()`
   *   afterwards (or from a running loop) and the promise resolves with the
   *   final post-processed output.
   * - `readPixels(rt)` reads a RenderTarget immediately (it must have been
   *   rendered to at least once).
   */
  readPixels(target?: RenderTarget): Promise<PixelData> {
    if (!target) {
      return new Promise<PixelData>((resolve) => this.pendingCapture.push(resolve));
    }
    const res = this.renderTargets.get(target);
    if (!res) {
      return Promise.reject(new Error(
        'readPixels: this RenderTarget has never been rendered to. ' +
        'Call renderer.render(scene, camera, target) first.',
      ));
    }
    const w = target.width;
    const h = target.height;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const buffer = this.device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: res.color }, { buffer, bytesPerRow }, [w, h]);
    this.device.queue.submit([encoder.finish()]);
    return new Promise<PixelData>((resolve) => this.resolveCapture(buffer, w, h, false, [resolve]));
  }

  /**
   * Capture a PNG of the next canvas frame (or of a RenderTarget). Returns a
   * Blob — save it, upload it, or `URL.createObjectURL` it.
   */
  async screenshot(target?: RenderTarget): Promise<Blob> {
    const { data, width, height } = await this.readPixels(target);
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('screenshot: OffscreenCanvas 2D context unavailable.');
    ctx.putImageData(new ImageData(data, width, height), 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  /**
   * Capture and GPU-project every point in an irradiance grid to SH-L2.
   * The resulting CPU-side coefficients are immediately serializable and are
   * uploaded lazily on the next render.
   */
  async bakeIrradianceProbes(
    grid: IrradianceProbeGrid,
    scene: Scene,
    options: IrradianceBakeOptions = {},
  ): Promise<Float32Array<ArrayBuffer>> {
    if (this.disposed || this.deviceLost) {
      throw new Error('bakeIrradianceProbes: renderer device is unavailable.');
    }
    scene.updateMatrixWorld();
    grid.updateWorldMatrix(true, false);
    const probe = new ReflectionProbe({
      resolution: options.resolution ?? 64,
      near: options.near ?? 0.1,
      far: options.far ?? 1000,
    });
    const coefficients = new Float32Array(grid.probeCount * 9 * 4);
    const output = this.device.createBuffer({
      label: 'irradiance-probe-output', size: 9 * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: 'irradiance-probe-readback', size: 9 * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.ensureIrradianceProbePipeline();
    const [nx, ny, nz] = grid.dimensions;
    const origin = grid.getWorldPosition(new Vector3());
    let index = 0;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++, index++) {
          probe.position.set(
            origin.x + x * grid.spacing.x,
            origin.y + y * grid.spacing.y,
            origin.z + z * grid.spacing.z,
          );
          probe.updateMatrixWorld();
          this.captureReflectionProbe(probe, scene, 0, false, false);
          const resources = this.reflectionProbeResources.get(probe)!;
          const bindGroup = this.device.createBindGroup({
            layout: this.irradianceProbeLayout!,
            entries: [
              { binding: 0, resource: resources.equirectView },
              { binding: 1, resource: this.reflectionProbeSampler },
              { binding: 2, resource: { buffer: output } },
            ],
          });
          const encoder = this.device.createCommandEncoder({ label: 'irradiance-probe-project' });
          const pass = encoder.beginComputePass();
          pass.setPipeline(this.irradianceProbePipeline!);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(1);
          pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, 9 * 16);
          this.device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          coefficients.set(new Float32Array(readback.getMappedRange()).slice(), index * 9 * 4);
          readback.unmap();
        }
      }
    }
    output.destroy();
    readback.destroy();
    grid.setCoefficients(coefficients);
    return coefficients;
  }

  /**
   * Lazily allocate a RenderTarget's GPU resources (color + depth + MSAA) and
   * register its sampleable view with the TextureManager so `rt.texture` works
   * in any material.
   */
  private getRenderTargetResources(rt: RenderTarget): RenderTargetResources {
    let res = this.renderTargets.get(rt);
    if (!res) {
      const color = this.device.createTexture({
        label: 'render-target',
        size: [rt.width, rt.height],
        format: RT_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        viewFormats: ['rgba8unorm-srgb'],
      });
      const depth = this.device.createTexture({
        label: 'render-target-depth',
        size: [rt.width, rt.height],
        format: DEPTH_FORMAT,
        sampleCount: this.sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      let msaaView: GPUTextureView | null = null;
      if (this.sampleCount > 1) {
        msaaView = this.device.createTexture({
          label: 'render-target-msaa',
          size: [rt.width, rt.height],
          format: RT_FORMAT,
          sampleCount: this.sampleCount,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        }).createView();
      }
      const sampler = this.device.createSampler({
        magFilter: 'linear', minFilter: 'linear',
        addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
      });
      // Materials sample through the -srgb view (decodes back to linear).
      this.textures.setExternal(rt.texture, {
        texture: color,
        view: color.createView({ format: 'rgba8unorm-srgb' }),
        sampler,
        version: rt.texture.version,
      });
      res = { color, renderView: color.createView(), msaaView, depthView: depth.createView() };
      this.renderTargets.set(rt, res);
    }
    return res;
  }

  private refreshReflectionProbes(scene: Scene): void {
    const probes: ReflectionProbe[] = [];
    scene.traverseVisible((object) => {
      if (object instanceof ReflectionProbe && probes.length < MAX_REFLECTION_PROBES) probes.push(object);
    });
    for (let slot = 0; slot < probes.length; slot++) {
      const probe = probes[slot];
      const resources = this.reflectionProbeResources.get(probe);
      const due = !resources || resources.slot !== slot || probe.needsUpdate ||
        (probe.refresh === 'every-n-frames' && this.frameNumber - resources.lastCaptureFrame >= probe.refreshInterval);
      if (due) this.captureReflectionProbe(probe, scene, slot);
    }
  }

  private captureReflectionProbe(
    probe: ReflectionProbe,
    scene: Scene,
    slot: number,
    copyToProbeArray = true,
    prefilter = true,
  ): void {
    const savedFrame = this.frameNumber;
    const savedLastRenderTime = this.lastRenderTime;
    const position = probe.getWorldPosition(new Vector3());
    this.capturingReflectionProbe = true;
    try {
      for (let face = 0; face < 6; face++) {
        const camera = probe.camera;
        camera.position.copy(position);
        camera.up.copy(PROBE_FACE_UPS[face]);
        camera.lookAt(position.clone().add(POINT_FACE_DIRS[face]));
        camera.updateMatrixWorld();
        this.render(scene, camera, probe.targets[face]);
      }
    } finally {
      this.capturingReflectionProbe = false;
      this.frameNumber = savedFrame;
      this.lastRenderTime = savedLastRenderTime;
    }

    let resources = this.reflectionProbeResources.get(probe);
    if (!resources) {
      const equirect = this.device.createTexture({
        label: 'reflection-probe-equirect', size: [256, 128], format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      resources = {
        equirect,
        equirectView: equirect.createView(),
        ibl: new IBLPrefilter(this.device),
        lastCaptureFrame: -1,
        slot,
      };
      this.reflectionProbeResources.set(probe, resources);
    }
    resources.slot = slot;
    this.ensureReflectionProbePipeline();
    const entries: GPUBindGroupEntry[] = probe.targets.map((target, binding) => ({
      binding,
      resource: this.textures.get(target.texture).view,
    }));
    entries.push({ binding: 6, resource: this.reflectionProbeSampler });
    entries.push({ binding: 7, resource: resources.equirectView });
    const bindGroup = this.device.createBindGroup({ layout: this.reflectionProbeLayout!, entries });
    const encoder = this.device.createCommandEncoder({ label: 'reflection-probe-capture' });
    const pass = encoder.beginComputePass({ label: 'cube-to-equirect' });
    pass.setPipeline(this.reflectionProbePipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(32, 16);
    pass.end();
    if (prefilter) resources.ibl.convolve(encoder, resources.equirectView);
    if (copyToProbeArray && prefilter) {
      for (let mip = 0; mip < IBL_MIP_LEVELS; mip++) {
        encoder.copyTextureToTexture(
          { texture: resources.ibl.prefiltered, mipLevel: mip },
          { texture: this.reflectionProbeTexture, mipLevel: mip, origin: { x: 0, y: 0, z: slot } },
          { width: Math.max(1, 256 >> mip), height: Math.max(1, 128 >> mip), depthOrArrayLayers: 1 },
        );
      }
    }
    this.device.queue.submit([encoder.finish()]);
    resources.lastCaptureFrame = savedFrame;
    probe.needsUpdate = false;
  }

  private ensureReflectionProbePipeline(): void {
    if (this.reflectionProbePipeline) return;
    const entries: GPUBindGroupLayoutEntry[] = [0, 1, 2, 3, 4, 5].map((binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      texture: { sampleType: 'float', viewDimension: '2d' },
    }));
    entries.push(
      { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
    );
    this.reflectionProbeLayout = this.device.createBindGroupLayout({
      label: 'reflection-probe-convert',
      entries,
    });
    this.reflectionProbePipeline = this.device.createComputePipeline({
      label: 'reflection-probe-convert',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.reflectionProbeLayout] }),
      compute: { module: this.device.createShaderModule({ code: REFLECTION_PROBE_SHADER }), entryPoint: 'cs_main' },
    });
  }

  private ensureIrradianceProbePipeline(): void {
    if (this.irradianceProbePipeline) return;
    this.irradianceProbeLayout = this.device.createBindGroupLayout({
      label: 'irradiance-probe-project',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.irradianceProbePipeline = this.device.createComputePipeline({
      label: 'irradiance-probe-project',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.irradianceProbeLayout] }),
      compute: { module: this.device.createShaderModule({ code: IRRADIANCE_PROBE_SHADER }), entryPoint: 'cs_main' },
    });
  }

  private prepareReflectionProbeBindings(scene: Scene): void {
    const data = this.reflectionProbeData;
    data.fill(0);
    let count = 0;
    scene.traverseVisible((object) => {
      if (!(object instanceof ReflectionProbe) || count >= MAX_REFLECTION_PROBES) return;
      const resources = this.reflectionProbeResources.get(object);
      if (!resources) return;
      const p = object.getWorldPosition(this._meshPos);
      const base = resources.slot * 4;
      data[base] = p.x; data[base + 1] = p.y; data[base + 2] = p.z; data[base + 3] = object.radius;
      data[16 + resources.slot] = object.intensity;
      count++;
    });
    data[20] = count;
    this.device.queue.writeBuffer(this.reflectionProbeBuffer, 0, data);
  }

  private prepareIrradianceGrid(scene: Scene): void {
    const grids: IrradianceProbeGrid[] = [];
    scene.traverseVisible((object) => {
      if (grids.length === 0 && object instanceof IrradianceProbeGrid && object.coefficients) grids.push(object);
    });
    const grid = grids[0];
    const params = new Float32Array(12);
    if (!grid || !grid.coefficients) {
      this.device.queue.writeBuffer(this.irradianceGridParams, 0, params);
      return;
    }
    const required = grid.coefficients.byteLength;
    const key = `${grid.id}:${grid.version}`;
    if (required > this.irradianceGridCapacity) {
      this.irradianceGridCapacity = Math.max(required, this.irradianceGridCapacity * 2);
      this.irradianceGridBuffer.destroy();
      this.irradianceGridBuffer = this.device.createBuffer({
        label: 'irradiance-grid-sh', size: this.irradianceGridCapacity,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.irradianceGridKey = '';
      this.buildFrameBindGroup();
    }
    if (key !== this.irradianceGridKey) {
      this.device.queue.writeBuffer(this.irradianceGridBuffer, 0, grid.coefficients);
      this.irradianceGridKey = key;
    }
    const p = grid.getWorldPosition(this._meshPos);
    params[0] = p.x; params[1] = p.y; params[2] = p.z; params[3] = 1;
    params[4] = grid.spacing.x; params[5] = grid.spacing.y; params[6] = grid.spacing.z;
    params[8] = grid.dimensions[0]; params[9] = grid.dimensions[1]; params[10] = grid.dimensions[2];
    this.device.queue.writeBuffer(this.irradianceGridParams, 0, params);
  }

  /**
   * Choose the shadow-casting directional light and fit an orthographic light
   * frustum to the opaque scene bounds, producing `_lightViewProj`. Sets
   * `shadowCasterIndex` (the light's index in the packed light array) or -1.
   */
  private prepareShadow(camera: Camera): void {
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
        const requestedCascades = Math.min(4, Math.max(1, Math.floor(this.shadowCascades)));
        if (requestedCascades > 1 && camera instanceof PerspectiveCamera) {
          this.prepareDirectionalCascades(camera, requestedCascades, minX, minY, minZ, maxX, maxY, maxZ);
        } else {
          this.cascadeCount = 1;
          this.cascadeViewProjs[0].copy(this._lightViewProj);
          this.cascadeSplits.fill(camera instanceof PerspectiveCamera ? camera.far : 1e30);
          this.uploadCascadeData();
        }
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

  private prepareDirectionalCascades(
    camera: PerspectiveCamera,
    count: number,
    sceneMinX: number, sceneMinY: number, sceneMinZ: number,
    sceneMaxX: number, sceneMaxY: number, sceneMaxZ: number,
  ): void {
    this.cascadeCount = count;
    const near = Math.max(camera.near, 0.001);
    const far = camera.far;
    computeCascadeSplits(near, far, count, this.shadowCascadeLambda, this.cascadeSplits);
    let previous = near;
    const tanHalfFov = Math.tan(camera.fov * Math.PI / 360);
    for (let cascade = 0; cascade < count; cascade++) {
      const split = this.cascadeSplits[cascade];

      for (let plane = 0; plane < 2; plane++) {
        const depth = plane === 0 ? previous : split;
        const halfH = tanHalfFov * depth;
        const halfW = halfH * camera.aspect;
        for (let corner = 0; corner < 4; corner++) {
          this.cascadeCorners[plane * 4 + corner]
            .set(corner & 1 ? halfW : -halfW, corner & 2 ? halfH : -halfH, -depth)
            .applyMatrix4(camera.matrixWorld);
        }
      }
      this._sceneCenter.set(0, 0, 0);
      for (const corner of this.cascadeCorners) this._sceneCenter.add(corner);
      this._sceneCenter.multiplyScalar(1 / 8);
      let radius = 0;
      for (const corner of this.cascadeCorners) radius = Math.max(radius, corner.distanceTo(this._sceneCenter));
      radius = Math.ceil(radius * 16) / 16;

      const eye = this._lightPos.set(
        this._sceneCenter.x - this._lightDir.x * radius * 2,
        this._sceneCenter.y - this._lightDir.y * radius * 2,
        this._sceneCenter.z - this._lightDir.z * radius * 2,
      );
      const up = Math.abs(this._lightDir.y) > 0.99 ? UNIT_Z : UNIT_Y;
      this._lightView.identity().lookAt(eye, this._sceneCenter, up).setPosition(eye).invert();

      let lminZ = Infinity, lmaxZ = -Infinity;
      for (let k = 0; k < 8; k++) {
        this._corner.set(
          k & 1 ? sceneMaxX : sceneMinX,
          k & 2 ? sceneMaxY : sceneMinY,
          k & 4 ? sceneMaxZ : sceneMinZ,
        ).applyMatrix4(this._lightView);
        lminZ = Math.min(lminZ, this._corner.z);
        lmaxZ = Math.max(lmaxZ, this._corner.z);
      }
      this._lightProj.makeOrthographic(-radius, radius, radius, -radius, -lmaxZ, -lminZ);
      this.cascadeViewProjs[cascade].multiplyMatrices(this._lightProj, this._lightView);
      previous = split;
    }
    for (let i = count; i < 4; i++) {
      this.cascadeViewProjs[i].copy(this.cascadeViewProjs[count - 1]);
    }
    this._lightViewProj.copy(this.cascadeViewProjs[0]);
    this.uploadCascadeData();
  }

  private uploadCascadeData(): void {
    const data = new Float32Array(72);
    for (let i = 0; i < 4; i++) data.set(this.cascadeViewProjs[i].elements, i * 16);
    data.set(this.cascadeSplits, 64);
    data[68] = this.cascadeCount;
    data[69] = Math.min(0.5, Math.max(0, this.shadowCascadeBlend));
    this.device.queue.writeBuffer(this.cascadeBuffer, 0, data);
    for (let i = 0; i < this.cascadeCount; i++) {
      this.device.queue.writeBuffer(this.cascadeLightBuffers[i], 0, new Float32Array(this.cascadeViewProjs[i].elements));
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
    } else if (scene.sky) {
      this.prepareProceduralSky(scene, encoder);
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

  /**
   * Generate the Preetham procedural sky into an equirect rgba16float texture
   * (when its parameters changed) and route it through the standard env path:
   * IBL prefilter + skybox. Param changes are detected by value, so mutating
   * `scene.sky` regenerates automatically — no flags.
   */
  private prepareProceduralSky(scene: Scene, encoder: GPUCommandEncoder): void {
    const sky = scene.sky!;
    const sd = sky.sunDirection;
    const turbidity = sky.turbidity ?? 4;
    const key = `sky:${sd.x.toFixed(4)},${sd.y.toFixed(4)},${sd.z.toFixed(4)},${turbidity}`;
    if (key !== this.envKey) {
      this.ensureSkyGenResources();
      const len = Math.hypot(sd.x, sd.y, sd.z) || 1;
      this.device.queue.writeBuffer(this.skyGenParamsBuffer!, 0,
        new Float32Array([sd.x / len, sd.y / len, sd.z / len, turbidity]));
      const pass = encoder.beginComputePass({ label: 'sky-gen' });
      pass.setPipeline(this.skyGenPipeline!);
      pass.setBindGroup(0, this.skyGenBindGroup!);
      pass.dispatchWorkgroups(SKY_TEX_WIDTH / 8, SKY_TEX_HEIGHT / 8);
      pass.end();

      this.envView = this.skyGenView!;
      this.envSampler = this.skyGenSampler!;
      this.envKey = key;
      // Re-convolve IBL from the fresh sky (encoded after sky-gen → ordered).
      this.ibl.convolve(encoder, this.skyGenView!);
      this.iblEnvKey = key;
      this.iblActive = true;
      this.buildFrameBindGroup();
    }
    this.envEnabled = true;
    this.envIntensity = scene.environmentIntensity;
    this.envMaxMip = IBL_MIP_LEVELS - 1;
  }

  private ensureSkyGenResources(): void {
    if (this.skyGenPipeline) return;
    const module = this.device.createShaderModule({ code: SKYGEN_SHADER, label: 'sky-gen' });
    const layout = this.device.createBindGroupLayout({
      label: 'sky-gen',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rgba16float', viewDimension: '2d', access: 'write-only' } },
      ],
    });
    this.skyGenPipeline = this.device.createComputePipeline({
      label: 'sky-gen',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'cs_sky' },
    });
    this.skyGenParamsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const tex = this.device.createTexture({
      label: 'procedural-sky',
      size: [SKY_TEX_WIDTH, SKY_TEX_HEIGHT],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.skyGenView = tex.createView();
    this.skyGenSampler = this.device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'repeat', addressModeV: 'clamp-to-edge',
    });
    this.skyGenBindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.skyGenParamsBuffer } },
        { binding: 1, resource: this.skyGenView },
      ],
    });
  }

  private renderShadowPass(encoder: GPUCommandEncoder): void {
    const tileSize = this.cascadeCount > 1 ? this.shadowMapSize / 2 : this.shadowMapSize;
    for (let cascade = 0; cascade < this.cascadeCount; cascade++) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowView,
          depthClearValue: 1.0,
          depthLoadOp: cascade === 0 ? 'clear' : 'load',
          depthStoreOp: 'store',
        },
      });
      if (this.cascadeCount > 1) {
        pass.setViewport((cascade % 2) * tileSize, Math.floor(cascade / 2) * tileSize, tileSize, tileSize, 0, 1);
        pass.setScissorRect((cascade % 2) * tileSize, Math.floor(cascade / 2) * tileSize, tileSize, tileSize);
      }
      pass.setPipeline(this.pipelines.shadowPipeline);
      pass.setBindGroup(0, this.cascadeLightBindGroups[cascade]);
      for (const mesh of this.opaque) {
        if (mesh instanceof InstancedMesh) continue; // instanced casters unsupported in v1
        const geometry = this.geometries.get(mesh.geometry);
        const slot = this.getMeshSlot(mesh);
        pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);
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
        const slot = this.getMeshSlot(mesh);
        pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);
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

  private prepareVolumetricFog(encoder: GPUCommandEncoder, scene: Scene, camera: Camera): void {
    if (!this.volumetricFog || !scene.fog) return;
    if (!this.volumetricFogPipeline) {
      this.volumetricFogLayout = this.device.createBindGroupLayout({
        label: 'volumetric-fog',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
          { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'comparison' } },
          { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      });
      this.volumetricFogPipeline = this.device.createComputePipeline({
        label: 'volumetric-fog',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.volumetricFogLayout] }),
        compute: { module: this.device.createShaderModule({ code: VOLUMETRIC_FOG_SHADER }), entryPoint: 'cs_main' },
      });
    }
    const clusterBuffer = this.clusterLightsBuffer ?? this.clusterDummyBuffer!;
    if (!this.volumetricFogBindGroup || this.volumetricFogClusterBuffer !== clusterBuffer) {
      this.volumetricFogBindGroup = this.device.createBindGroup({
        layout: this.volumetricFogLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.frameBuffer } },
          { binding: 1, resource: { buffer: this.lightBuffer } },
          { binding: 2, resource: { buffer: clusterBuffer } },
          { binding: 3, resource: { buffer: this.cascadeBuffer } },
          { binding: 4, resource: this.shadowView },
          { binding: 5, resource: this.shadowSampler },
          { binding: 6, resource: this.volumetricFogView },
          { binding: 7, resource: { buffer: this.volumetricFogInvVP } },
        ],
      });
      this.volumetricFogClusterBuffer = clusterBuffer;
    }
    this._inverseViewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).invert();
    this.device.queue.writeBuffer(this.volumetricFogInvVP, 0, new Float32Array(this._inverseViewProjection.elements));
    const pass = encoder.beginComputePass({ label: 'volumetric-fog' });
    pass.setPipeline(this.volumetricFogPipeline);
    pass.setBindGroup(0, this.volumetricFogBindGroup);
    pass.dispatchWorkgroups(Math.ceil(CLUSTER_X / 4), Math.ceil(CLUSTER_Y / 3), Math.ceil(CLUSTER_Z / 4));
    pass.end();
  }

  // Scratch buffers for per-frame particle uploads (no hot-loop allocation).
  private _particleSimData = new Float32Array(20);
  private _particleSimU32 = new Uint32Array(this._particleSimData.buffer);
  private _particleDrawData = new Float32Array(12);

  private ensureParticleSimResources(): void {
    if (this.particleSimPipeline) return;
    const module = this.device.createShaderModule({ code: PARTICLE_SIM_SHADER, label: 'particle-sim' });
    this.particleSimLayout = this.device.createBindGroupLayout({
      label: 'particle-sim',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.particleSimPipeline = this.device.createComputePipeline({
      label: 'particle-sim',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.particleSimLayout] }),
      compute: { module, entryPoint: 'cs_sim' },
    });
  }

  private getParticleResources(sys: ParticleSystem): ParticleResources {
    const capacity = sys.options.capacity ?? 1000;
    let res = this.particleResources.get(sys);
    if (!res || res.capacity !== capacity) {
      res?.stateBuffer.destroy();
      res?.simParams.destroy();
      res?.drawParams.destroy();
      const stateBuffer = this.device.createBuffer({
        size: capacity * 48, // Particle: 3 × vec4 (zero-initialized = all dead)
        usage: GPUBufferUsage.STORAGE,
      });
      const simParams = this.device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const drawParams = this.device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const simBindGroup = this.device.createBindGroup({
        layout: this.particleSimLayout!,
        entries: [
          { binding: 0, resource: { buffer: simParams } },
          { binding: 1, resource: { buffer: stateBuffer } },
        ],
      });
      const drawBindGroup = this.device.createBindGroup({
        layout: this.pipelines.particleLayout,
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: drawParams } },
        ],
      });
      res = { capacity, stateBuffer, simParams, simBindGroup, drawParams, drawBindGroup, cursor: 0, carry: 0 };
      this.particleResources.set(sys, res);
    }
    return res;
  }

  /** Upload per-system params and encode the particle integration compute pass. */
  private prepareParticles(encoder: GPUCommandEncoder, dt: number): void {
    if (this.particleSystems.length === 0) return;
    this.ensureParticleSimResources();

    const f32 = this._particleSimData;
    const u32 = this._particleSimU32;
    const dp = this._particleDrawData;
    for (const sys of this.particleSystems) {
      const res = this.getParticleResources(sys);
      const o = sys.options;

      // Emission budget: rate × dt with fractional carry, ring-cursor advance.
      let emit = 0;
      if (sys.emitting && dt > 0) {
        res.carry += (o.rate ?? 100) * dt;
        emit = Math.min(Math.floor(res.carry), res.capacity);
        res.carry = Math.min(res.carry - emit, 1);
      }

      sys.getWorldPosition(this._meshPos);
      const life = o.lifetime ?? 2;
      const [lifeMin, lifeMax] = typeof life === 'number' ? [life, life] : life;
      f32[0] = this._meshPos.x; f32[1] = this._meshPos.y; f32[2] = this._meshPos.z;
      f32[3] = dt;
      f32[4] = o.velocity?.x ?? 0; f32[5] = o.velocity?.y ?? 1; f32[6] = o.velocity?.z ?? 0;
      f32[7] = o.spread ?? 0.5;
      f32[8] = o.gravity?.x ?? 0; f32[9] = o.gravity?.y ?? 0; f32[10] = o.gravity?.z ?? 0;
      f32[11] = this._elapsed; // randomness seed (pinned by renderer.time when deterministic)
      f32[12] = lifeMin; f32[13] = lifeMax; f32[14] = 0; f32[15] = 0;
      u32[16] = res.cursor; u32[17] = emit; u32[18] = res.capacity; u32[19] = 0;
      this.device.queue.writeBuffer(res.simParams, 0, f32);
      res.cursor = (res.cursor + emit) % res.capacity;

      // Draw params: size/color/opacity over life (re-read every frame, no flags).
      const size = o.size ?? 0.1;
      const [s0, s1] = typeof size === 'number' ? [size, size] : size;
      const c0 = Array.isArray(o.color) ? o.color[0] : o.color;
      const c1 = Array.isArray(o.color) ? o.color[1] : o.color;
      const op = o.opacity ?? [1, 0];
      const [o0, o1] = typeof op === 'number' ? [op, op] : op;
      dp[0] = s0; dp[1] = s1; dp[2] = 0; dp[3] = 0;
      dp[4] = c0?.r ?? 1; dp[5] = c0?.g ?? 1; dp[6] = c0?.b ?? 1; dp[7] = o0;
      dp[8] = c1?.r ?? 1; dp[9] = c1?.g ?? 1; dp[10] = c1?.b ?? 1; dp[11] = o1;
      this.device.queue.writeBuffer(res.drawParams, 0, dp);
    }

    const pass = encoder.beginComputePass({ label: 'particle-sim' });
    pass.setPipeline(this.particleSimPipeline!);
    for (const sys of this.particleSystems) {
      const res = this.particleResources.get(sys)!;
      pass.setBindGroup(0, res.simBindGroup);
      pass.dispatchWorkgroups(Math.ceil(res.capacity / 64));
    }
    pass.end();
  }

  /** Draw all particle systems as instanced billboards (inside the scene pass). */
  private drawParticles(pass: GPURenderPassEncoder): void {
    for (const sys of this.particleSystems) {
      const res = this.particleResources.get(sys);
      if (!res) continue;
      const additive = (sys.options.blending ?? 'additive') === 'additive';
      pass.setPipeline(this.pipelines.getParticles(this.sceneTargetFormat, additive, this._sceneInputs));
      pass.setBindGroup(0, this.frameBindGroup);
      pass.setBindGroup(1, res.drawBindGroup);
      pass.draw(6, res.capacity);
    }
  }

  /**
   * Get (or create/grow) the batch for a (texture, screen, sdf) combination
   * with room for one more instance. Bind groups rebuild when the buffer grows
   * or the texture version changes.
   */
  private getSpriteBatch(key: string, screen: boolean, sdf: boolean, textureSig: string, view: GPUTextureView, sampler: GPUSampler): SpriteBatch {
    let batch = this.spriteBatches.get(key);
    const needsGrow = batch !== undefined && batch.count >= batch.capacity;
    if (!batch || needsGrow || batch.textureSig !== textureSig) {
      const capacity = needsGrow ? batch!.capacity * 2 : (batch?.capacity ?? 16);
      const buffer = needsGrow || !batch
        ? this.device.createBuffer({ size: capacity * SPRITE_STRIDE_F * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
        : batch.buffer;
      if (needsGrow) batch!.buffer.destroy();
      let params = batch?.params;
      if (!params) {
        params = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(params, 0, new Float32Array([screen ? 1 : 0, sdf ? 1 : 0, 0, 0]));
      }
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.spriteLayout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: params } },
          { binding: 2, resource: view },
          { binding: 3, resource: sampler },
        ],
      });
      const data = needsGrow
        ? (() => { const d = new Float32Array(capacity * SPRITE_STRIDE_F); d.set(batch!.data); return d; })()
        : (batch?.data ?? new Float32Array(capacity * SPRITE_STRIDE_F));
      batch = { screen, sdf, textureSig, data, capacity, count: batch?.count ?? 0, buffer, params, bindGroup };
      this.spriteBatches.set(key, batch);
    }
    return batch;
  }

  /** Rebuild sprite batches from the collected sprites and upload instance data. */
  private prepareSprites(): void {
    if (this.sprites.length === 0 && this.texts.length === 0 && this.spriteBatches.size === 0) return;
    for (const batch of this.spriteBatches.values()) batch.count = 0;

    for (const sprite of this.sprites) {
      const tex = sprite.texture;
      const entry = tex ? this.textures.get(tex) : null;
      const textureSig = tex ? `${tex.id}:${tex.version}` : 'white';
      const screen = sprite.screenSpace;
      const key = `${tex ? tex.id : 'white'}|${screen ? 1 : 0}|0`;
      const batch = this.getSpriteBatch(
        key, screen, false, textureSig,
        entry ? entry.view : this.textures.defaultWhiteView,
        entry ? entry.sampler : this.textures.defaultSampler,
      );
      sprite.getWorldPosition(this._meshPos);
      const px = screen ? this.pixelRatio : 1; // CSS px → device px
      const d = batch.data;
      const i = batch.count * SPRITE_STRIDE_F;
      d[i] = this._meshPos.x; d[i + 1] = this._meshPos.y; d[i + 2] = this._meshPos.z; d[i + 3] = 0;
      d[i + 4] = sprite.size.x * px; d[i + 5] = sprite.size.y * px;
      d[i + 6] = sprite.offset.x * px; d[i + 7] = sprite.offset.y * px;
      d[i + 8] = sprite.color.r; d[i + 9] = sprite.color.g; d[i + 10] = sprite.color.b;
      d[i + 11] = sprite.opacity;
      d[i + 12] = 0; d[i + 13] = 0; d[i + 14] = 1; d[i + 15] = 1; // full texture
      batch.count++;
    }

    // SDF text: one glyph instance per character, batched per font atlas.
    for (const tm of this.texts) {
      if (!tm.text) continue;
      let atlas = this.fontAtlases.get(tm.font);
      if (!atlas) {
        atlas = new SDFFontAtlas(tm.font);
        this.fontAtlases.set(tm.font, atlas);
      }
      // Rasterize any new glyphs first (bumps the atlas texture version), so
      // textures.get() below uploads the finished atlas.
      for (const ch of tm.text) {
        if (ch !== '\n') atlas.getGlyph(ch);
      }
      const entry = this.textures.get(atlas.texture);
      const textureSig = `${atlas.texture.id}:${atlas.texture.version}`;
      const screen = tm.screenSpace;
      const key = `sdf:${tm.font}|${screen ? 1 : 0}|1`;
      tm.getWorldPosition(this._meshPos);
      const scale = (tm.fontSize / SDF_BASE_FONT) * (screen ? this.pixelRatio : 1);

      let lineY = 0;
      for (const line of tm.text.split('\n')) {
        let width = 0;
        for (const ch of line) width += atlas.advanceOf(ch);
        let pen = tm.anchor === 'left' ? 0 : tm.anchor === 'center' ? -width / 2 : -width;
        for (const ch of line) {
          const g = atlas.getGlyph(ch);
          if (g) {
            const batch = this.getSpriteBatch(key, screen, true, textureSig, entry.view, entry.sampler);
            const d = batch.data;
            const i = batch.count * SPRITE_STRIDE_F;
            d[i] = this._meshPos.x; d[i + 1] = this._meshPos.y; d[i + 2] = this._meshPos.z; d[i + 3] = 0;
            d[i + 4] = g.w * scale; d[i + 5] = g.h * scale;
            d[i + 6] = (pen + g.cx) * scale; d[i + 7] = (lineY + g.cy) * scale;
            d[i + 8] = tm.color.r; d[i + 9] = tm.color.g; d[i + 10] = tm.color.b;
            d[i + 11] = tm.opacity;
            d[i + 12] = g.u0; d[i + 13] = g.v0; d[i + 14] = g.u1; d[i + 15] = g.v1;
            batch.count++;
          }
          pen += g ? g.advance : atlas.advanceOf(ch);
        }
        lineY -= atlas.lineHeight;
      }
    }

    // Upload live batches; drop batches that went empty (texture/mode changed).
    for (const [key, batch] of this.spriteBatches) {
      if (batch.count === 0) {
        batch.buffer.destroy();
        batch.params.destroy();
        this.spriteBatches.delete(key);
      } else {
        this.device.queue.writeBuffer(batch.buffer, 0, batch.data, 0, batch.count * SPRITE_STRIDE_F);
      }
    }
  }

  /** Draw sprite batches: world-space first (depth-tested), then HUD overlays. */
  private drawSprites(pass: GPURenderPassEncoder): void {
    for (const screenPhase of [false, true]) {
      for (const batch of this.spriteBatches.values()) {
        if (batch.count === 0 || batch.screen !== screenPhase) continue;
        pass.setPipeline(this.pipelines.getSprites(this.sceneTargetFormat, batch.screen, this._sceneInputs));
        pass.setBindGroup(0, this.frameBindGroup);
        pass.setBindGroup(1, batch.bindGroup);
        pass.draw(6, batch.count);
      }
    }
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
          { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
          { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'non-filtering' } },
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

    // Occlusion is active only when opted in, MSAA is off, and a prior-frame
    // pyramid exists. The hi-Z and its matching viewProj come from last frame.
    const occlusion = this.occlusionCulling && this.sampleCount === 1 && this.hizValid && !!this.hizSampleView;
    this.ensureHizDummy();

    // Upload frustum planes + prev viewProj + draw count + hi-Z params.
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
    pf.set(this._prevHizViewProj, 24);  // prevViewProj at byte 96 (float 24)
    pu[40] = this.nextMeshSlot;          // drawCount at byte 160
    pu[41] = occlusion ? 1 : 0;          // occlusion flag
    pu[42] = this.hizMipCount;           // hizMips
    pf[44] = this.canvas.width;          // hizSize.x at byte 176
    pf[45] = this.canvas.height;         // hizSize.y
    this.device.queue.writeBuffer(this.gpuCullParamsBuffer!, 0, paramsBuf);

    // Rebuild bind group when buffers or the bound hi-Z view change.
    const hizView = occlusion ? this.hizSampleView! : this.hizDummyView!;
    if (!this.gpuCullBindGroup || this._boundHizView !== hizView) {
      this.gpuCullBindGroup = this.device.createBindGroup({
        layout: this.gpuCullLayout!,
        entries: [
          { binding: 0, resource: { buffer: this.gpuCullParamsBuffer! } },
          { binding: 1, resource: { buffer: this.gpuSphereBuffer! } },
          { binding: 2, resource: { buffer: this.gpuIndirectBuffer! } },
          { binding: 3, resource: hizView },
          { binding: 4, resource: this.hizSampler! },
        ],
      });
      this._boundHizView = hizView;
    }

    const pass = encoder.beginComputePass({ label: 'frustum-cull' });
    pass.setPipeline(this.gpuCullPipeline!);
    pass.setBindGroup(0, this.gpuCullBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nextMeshSlot / 64));
    pass.end();
  }

  /** Lazily create the 1×1 dummy hi-Z (bound when no real pyramid is active). */
  private ensureHizDummy(): void {
    if (!this.hizSampler) {
      this.hizSampler = this.device.createSampler({ label: 'hiz', magFilter: 'nearest', minFilter: 'nearest' });
    }
    if (!this.hizDummyView) {
      const tex = this.device.createTexture({
        label: 'hiz-dummy', size: [1, 1], format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.hizDummyView = tex.createView();
    }
  }

  /** (Re)allocate the hi-Z pyramid for the current canvas size + its pipelines. */
  private ensureHizResources(): void {
    const w = this.canvas.width, h = this.canvas.height;
    const mips = Math.max(1, Math.floor(Math.log2(Math.max(w, h))) + 1);
    if (this.hizTexture && this.hizMipCount === mips &&
        this.hizTexture.width === w && this.hizTexture.height === h) {
      return;
    }
    this.hizTexture?.destroy();
    this.hizTexture = this.device.createTexture({
      label: 'hiz',
      size: [w, h],
      format: 'r32float',
      mipLevelCount: mips,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hizMipViews = [];
    for (let i = 0; i < mips; i++) {
      this.hizMipViews.push(this.hizTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }));
    }
    this.hizSampleView = this.hizTexture.createView(); // full chain, for cull sampling
    this.hizMipCount = mips;
    this.hizValid = false; // stale until rebuilt
    this._boundHizView = null;

    if (!this.hizCopyPipeline) {
      const copyModule = this.device.createShaderModule({ code: HIZ_COPY_SHADER, label: 'hiz-copy' });
      const downModule = this.device.createShaderModule({ code: HIZ_DOWN_SHADER, label: 'hiz-down' });
      this.hizCopyLayout = this.device.createBindGroupLayout({
        label: 'hiz-copy',
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } }],
      });
      this.hizDownLayout = this.device.createBindGroupLayout({
        label: 'hiz-down',
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } }],
      });
      this.hizCopyPipeline = this.device.createRenderPipeline({
        label: 'hiz-copy',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.hizCopyLayout] }),
        vertex: { module: copyModule, entryPoint: 'vs_main' },
        fragment: { module: copyModule, entryPoint: 'fs_main', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-list' },
      });
      this.hizDownPipeline = this.device.createRenderPipeline({
        label: 'hiz-down',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.hizDownLayout] }),
        vertex: { module: downModule, entryPoint: 'vs_main' },
        fragment: { module: downModule, entryPoint: 'fs_main', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-list' },
      });
    }
  }

  /**
   * Build the hi-Z max-depth pyramid from this frame's depth (mip 0 = copy of
   * depth, each finer→coarser mip = 2×2 max). Runs after the main pass; the
   * result is consumed immediately by SSR and by next frame's cull. Records the
   * viewProj it was built with so the cull can reproject against it.
   */
  private buildHiZ(encoder: GPUCommandEncoder): void {
    if (this.sampleCount !== 1 || !this.depthSampleView) return;
    this.ensureHizResources();
    this.ensureHizDummy();

    // Mip 0: copy scene depth.
    const copyBind = this.device.createBindGroup({
      layout: this.hizCopyLayout!,
      entries: [{ binding: 0, resource: this.depthSampleView }],
    });
    let p = encoder.beginRenderPass({
      colorAttachments: [{ view: this.hizMipViews[0], loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 0, b: 0, a: 1 } }],
    });
    p.setPipeline(this.hizCopyPipeline!);
    p.setBindGroup(0, copyBind);
    p.draw(3);
    p.end();

    // Coarser mips: 2×2 max-reduce the previous level.
    for (let i = 1; i < this.hizMipCount; i++) {
      const downBind = this.device.createBindGroup({
        layout: this.hizDownLayout!,
        entries: [{ binding: 0, resource: this.hizMipViews[i - 1] }],
      });
      const dp = encoder.beginRenderPass({
        colorAttachments: [{ view: this.hizMipViews[i], loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 0, b: 0, a: 1 } }],
      });
      dp.setPipeline(this.hizDownPipeline!);
      dp.setBindGroup(0, downBind);
      dp.draw(3);
      dp.end();
    }

    // This pyramid matches the current frame's viewProj; the cull next frame
    // reads it as "prev".
    this._prevHizViewProj.set(this._viewProjection.elements);
    this.hizValid = true;
  }

  private clearColor(scene: Scene): GPUColor {
    const bg = scene.background;
    if (!bg) return { r: 0.05, g: 0.05, b: 0.06, a: 1 };
    if (this._usePost) return { r: bg.r, g: bg.g, b: bg.b, a: 1 };
    // background is linear; encode to sRGB-ish for the non-srgb target
    const enc = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
    return { r: enc(bg.r), g: enc(bg.g), b: enc(bg.b), a: 1 };
  }

  private collect(scene: Scene): void {
    this.opaque.length = 0;
    this.transparent.length = 0;
    this.lights.length = 0;
    this.particleSystems.length = 0;
    this.sprites.length = 0;
    this.texts.length = 0;
    this.culledCount = 0;
    this.hasShells = false;

    scene.traverseVisible((object: Object3D) => {
      if (object instanceof Light) {
        this.lights.push(object);
      } else if (object instanceof ParticleSystem) {
        this.particleSystems.push(object);
      } else if (object instanceof Sprite) {
        this.sprites.push(object);
      } else if (object instanceof TextMesh) {
        this.texts.push(object);
      } else if (object instanceof Mesh) {
        if (this.frustumCulling && object.frustumCulled && this.isCulled(object)) {
          this.culledCount++;
          return;
        }
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        const transparent = materials.some((m) => m.transparent);
        if (transparent) this.transparent.push(object);
        else this.opaque.push(object);
        if (object.shell) this.hasShells = true;
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
    // Bit 0: linear output; bit 1: IBL; bit 2: screen-space scene capture.
    // _usePost is false for render-target passes (direct pipeline, in-shader tonemap).
    f[63] = (this._usePost ? 1 : 0) |
      (this.iblActive ? 2 : 0) |
      (this._usePost ? 4 : 0);

    // clusterParams (64..67) + clusterDims (68..71)
    const cam = camera as unknown as { near?: number; far?: number };
    f[64] = this.clusteredLighting ? 1 : 0;
    f[65] = Math.max(cam.near ?? 0.1, 0.001);
    f[66] = cam.far ?? 2000;
    f[67] = scene.backgroundBlur; // skybox blur (0 = sharp, 1 = max mip)
    f[68] = this._renderWidth / CLUSTER_X;  // tile size in pixels
    f[69] = this._renderHeight / CLUSTER_Y;
    f[70] = this._elapsed; // seconds for shader elapsedTime() (renderer.time when deterministic)
    f[71] = 0;

    // fogColor (72..75) + fogParams (76..79)
    const fog = scene.fog;
    if (fog) {
      const exp2 = fog.density !== undefined;
      f[72] = fog.color.r;
      f[73] = fog.color.g;
      f[74] = fog.color.b;
      f[75] = exp2 ? 2 : 1; // mode
      f[76] = exp2 ? fog.density! : (fog.near ?? 1);
      f[77] = exp2 ? 0 : (fog.far ?? 100);
      f[78] = fog.heightFalloff ?? 0;
      f[79] = this.volumetricFog ? 1 : 0;
    } else {
      f[75] = 0; // mode = none
      f[79] = 0;
    }

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
      const hasBuffer = res.layout.fields.length > 0;
      const textureCount = res.layout.textures.length;
      pass.setPipeline(this.pipelines.getCustom(
        cacheKey, variant, sm, this.sceneTargetFormat, oit, oitSampleCount,
        () => buildSurfaceShader(variant, res.layout.wgsl, sm.surfaceCode, sm.vertexCode, sm.lightCode, sm.ambientCode),
        hasBuffer, textureCount,
        sm.name || sm.type,
        undefined,
        this._sceneInputs && !oit,
      ));
      pass.setBindGroup(2, res.bindGroup);
    } else {
      const std = material as StandardMaterial;
      pass.setPipeline(oit
        ? this.pipelines.getOIT(std, variant, oitSampleCount)
        : this.pipelines.get(std, variant, this.sceneTargetFormat, undefined, this._sceneInputs));
      pass.setBindGroup(2, this.getMaterialResources(std).bindGroup);
    }

    // Group 1: model uniform (static/skinned/morph) or instance storage (instanced)
    if (instanced) {
      pass.setBindGroup(1, this.getInstancedResources(mesh as InstancedMesh).bindGroup);
    } else {
      const slot = this.getMeshSlot(mesh);
      pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);
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

  /**
   * Inverted-hull shell: re-draw the mesh's geometry with back faces (front
   * cull), extruded along the normal by `shell.thickness`, using the shell
   * material. The standard PBR/Surface fragment shades it, so the look comes
   * entirely from `shell.material`. Drawn after the opaque pass so the depth
   * test leaves only the silhouette ring visible. InstancedMesh is unsupported.
   */
  private drawShell(pass: DrawEncoder, mesh: Mesh): void {
    const shell = mesh.shell;
    if (!shell || shell.thickness === 0 || mesh instanceof InstancedMesh) return;
    const material = shell.material;
    const isShader = material instanceof ShaderMaterial;
    if (!isShader && !(material instanceof StandardMaterial)) return;
    const geometry = this.geometries.get(mesh.geometry);

    const skinned = mesh instanceof SkinnedMesh && geometry.joints !== null && geometry.weights !== null;
    const morphed =
      !skinned &&
      mesh.morphTargetInfluences.length > 0 &&
      !!mesh.geometry.morphAttributes.position?.length;
    const variant = skinned ? 'skinned' : morphed ? 'morph' : 'static';

    if (isShader) {
      const sm = material as ShaderMaterial;
      const res = this.getShaderMaterialResources(sm);
      const cacheKey = `sm:${sm.id}:v${sm.version}:${res.shapeKey}`;
      pass.setPipeline(this.pipelines.getCustom(
        cacheKey, variant, sm, this.sceneTargetFormat, false, 1,
        () => buildSurfaceShader(variant, res.layout.wgsl, sm.surfaceCode, sm.vertexCode, sm.lightCode, sm.ambientCode),
        res.layout.fields.length > 0, res.layout.textures.length,
        sm.name || sm.type, 'front', this._sceneInputs,
      ));
      pass.setBindGroup(2, res.bindGroup);
    } else {
      const std = material as StandardMaterial;
      pass.setPipeline(this.pipelines.get(std, variant, this.sceneTargetFormat, 'front', this._sceneInputs));
      pass.setBindGroup(2, this.getMaterialResources(std).bindGroup);
    }

    const slot = this.getMeshShellSlot(mesh, shell.thickness);
    pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);

    pass.setVertexBuffer(0, geometry.position);
    pass.setVertexBuffer(1, geometry.normal);
    pass.setVertexBuffer(2, geometry.uv);
    pass.setVertexBuffer(3, geometry.tangent);
    if (skinned) {
      pass.setBindGroup(3, this.getSkinnedResources(mesh as SkinnedMesh).bindGroup);
      pass.setVertexBuffer(4, geometry.joints!);
      pass.setVertexBuffer(5, geometry.weights!);
      pass.setVertexBuffer(6, geometry.color);
    } else {
      pass.setVertexBuffer(4, geometry.color);
      if (morphed) pass.setBindGroup(3, this.getMorphResources(mesh).bindGroup);
    }

    if (geometry.index) {
      pass.setIndexBuffer(geometry.index, geometry.indexFormat);
      pass.drawIndexed(geometry.drawCount);
    } else {
      pass.draw(geometry.drawCount);
    }
  }

  private drawLine(pass: DrawEncoder, mesh: Mesh, material: LineBasicMaterial): void {
    // The color stream is always present (white default), so lines need no setup.
    const geometry = this.geometries.get(mesh.geometry);

    pass.setPipeline(this.pipelines.getLine(material, this.sceneTargetFormat, this._sceneInputs));
    const slot = this.getMeshSlot(mesh);
    pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);
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
    let key = `${colorFormat}|si${this._sceneInputs ? 1 : 0}|${this.frameBindGroupVersion}|`;
    for (const mesh of this.opaque) {
      const m = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      key += `${mesh.id},${mesh.geometry.id},${(m as Material).id}`;
      // ShaderMaterial recompiles invalidate the recorded pipeline.
      if (m instanceof ShaderMaterial) key += `v${m.version}`;
      key += ';';
    }
    if (this.opaqueBundle && key === this.bundleKey) return this.opaqueBundle;

    const encoder = this.device.createRenderBundleEncoder({
      colorFormats: this._sceneInputs ? [colorFormat, 'rgba16float'] : [colorFormat],
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
    this.writeModelSlot(slot, mesh, 0);
    return slot;
  }

  /**
   * Assign a stable pool slot for a mesh's shell draw and upload its model
   * matrix with the shell thickness in params.x. A separate slot from the main
   * draw, since both are encoded in one frame (the main draw needs thickness 0).
   */
  private getMeshShellSlot(mesh: Mesh, thickness: number): number {
    let slot = this.shellSlots.get(mesh);
    if (slot === undefined) {
      slot = this.nextMeshSlot++;
      this.shellSlots.set(mesh, slot);
    }
    this.writeModelSlot(slot, mesh, thickness);
    return slot;
  }

  /** Pack a mesh's world/normal matrix + shell thickness into a model-pool slot. */
  private writeModelSlot(slot: number, mesh: Mesh, thickness: number): void {
    this.ensureModelPool(slot + 1);
    const data = new Float32Array(MODEL_SIZE / 4);
    data.set(mesh.matrixWorld.elements, 0);
    this._normalMatrix.getNormalMatrix(mesh.matrixWorld);
    const nm = this._normalMatrix.elements;
    data[16] = nm[0]; data[17] = nm[1]; data[18] = nm[2]; data[19] = 0;
    data[20] = nm[3]; data[21] = nm[4]; data[22] = nm[5]; data[23] = 0;
    data[24] = nm[6]; data[25] = nm[7]; data[26] = nm[8]; data[27] = 0;
    data[28] = 0; data[29] = 0; data[30] = 0; data[31] = 1;
    data[32] = thickness; // params.x
    this.device.queue.writeBuffer(this.modelPoolBuffer!, slot * 256, data);
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
   * Uniform buffer + bind group for a ShaderMaterial. The buffer/bind group is
   * (re)built when the uniforms object's shape (scalar names+types and texture
   * names) changes — which also bumps the pipeline cache key, recompiling the
   * shader to match. The bind group also rebuilds when a texture's identity or
   * version changes. Scalar values are packed and uploaded once per frame.
   */
  private getShaderMaterialResources(material: ShaderMaterial): ShaderMaterialResources {
    const layout = computeUniformLayout(material.uniforms);
    const shapeKey = layout.fields.map((f) => `${f.name}:${f.kind}`).join(',') +
      '|tex:' + layout.textures.join(',');
    // Texture identity+version signature: rebuild the bind group when it changes.
    let textureSig = '';
    for (const name of layout.textures) {
      const tex = material.uniforms[name] as Texture;
      textureSig += `${name}=${tex.id}:${tex.version};`;
    }

    let res = this.shaderMaterialResources.get(material);
    if (!res || res.shapeKey !== shapeKey || res.textureSig !== textureSig) {
      const reshape = !res || res.shapeKey !== shapeKey;
      let buffer = res?.buffer ?? null;
      if (reshape) {
        res?.buffer?.destroy();
        buffer = layout.fields.length > 0
          ? this.device.createBuffer({ size: layout.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
          : null;
      }
      const entries: GPUBindGroupEntry[] = [];
      if (buffer) entries.push({ binding: 0, resource: { buffer } });
      layout.textures.forEach((name, i) => {
        const entry = this.textures.get(material.uniforms[name] as Texture);
        entries.push({ binding: 1 + i * 2, resource: entry.view });
        entries.push({ binding: 2 + i * 2, resource: entry.sampler });
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.customUniformLayout(layout.fields.length > 0, layout.textures.length),
        entries,
      });
      res = {
        buffer, bindGroup, data: new Float32Array(layout.size / 4),
        layout, shapeKey, uploadedFrame: -1, textureSig,
      };
      this.shaderMaterialResources.set(material, res);
    }
    if (res.buffer && res.uploadedFrame !== this.frameNumber) {
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
    const slot = this.getMeshSlot(mesh);
    pass.setBindGroup(1, this.modelPoolBindGroup!, [slot * 256]);
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
