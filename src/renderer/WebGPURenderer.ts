import type { Scene } from '../core/Scene';
import type { Camera } from '../core/Camera';
import type { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { Light } from '../lights/Light';
import { AmbientLight } from '../lights/AmbientLight';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { SkinnedMesh } from '../core/SkinnedMesh';
import { InstancedMesh } from '../core/InstancedMesh';
import type { Material } from '../materials/Material';
import { StandardMaterial } from '../materials/StandardMaterial';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
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
import { SHADOW_DEPTH_FORMAT } from './constants';
import { PostProcessing } from './PostProcessing';

const MAX_LIGHTS = 32;
const UNIT_Y = new Vector3(0, 1, 0);
const UNIT_Z = new Vector3(0, 0, 1);
const FRAME_SIZE = 256; // bytes (lightViewProj mat4 + shadowParams + envParams vec4s)
const MODEL_SIZE = 128;
const MATERIAL_SIZE = 144;
const LIGHT_STRIDE = 48; // bytes per light

interface MeshResources {
  modelBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
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

  // Post-processing (opt-in): render to an HDR target, then tonemap (+ FXAA).
  /** Route rendering through the HDR post pipeline (tonemap moves to a final pass). */
  postProcessing = false;
  /** Apply FXAA in the post pipeline (only when `postProcessing` is on). */
  fxaa = true;
  /** Apply bloom in the post pipeline (only when `postProcessing` is on). */
  bloom = false;
  bloomThreshold = 1.0;
  bloomIntensity = 0.6;
  private post!: PostProcessing;
  private _lightView = new Matrix4();
  private _lightProj = new Matrix4();
  private _lightViewProj = new Matrix4();
  private _lightPos = new Vector3();
  private _lightDir = new Vector3();
  private _sceneCenter = new Vector3();
  private _corner = new Vector3();
  private shadowCasterIndex = -1;

  private meshResources = new WeakMap<Mesh, MeshResources>();
  private materialResources = new WeakMap<Material, MaterialResources>();
  private skinnedResources = new WeakMap<SkinnedMesh, SkinnedResources>();
  private instancedResources = new WeakMap<InstancedMesh, InstancedResources>();
  private morphResources = new WeakMap<Mesh, MorphResources>();
  private lineResources = new WeakMap<LineBasicMaterial, { buffer: GPUBuffer; bindGroup: GPUBindGroup }>();

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
    this.buildFrameBindGroup();
  }

  private buildFrameBindGroup(): void {
    this.frameBindGroup = this.device.createBindGroup({
      layout: this.pipelines.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: this.shadowView },
        { binding: 3, resource: this.shadowSampler },
        { binding: 4, resource: this.envView },
        { binding: 5, resource: this.envSampler },
      ],
    });
  }

  /** Allocate the shadow map (sized to `shadowMapSize` when enabled, else 1×1). */
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
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: DEPTH_FORMAT,
      sampleCount: this.sampleCount,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

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

    // Build the view frustum for culling (projection * view).
    this._viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this._frustum.setFromProjectionMatrix(this._viewProjection);

    this.collect(scene);
    this.prepareShadow();
    this.prepareEnvironment(scene);
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

    // Shadow depth pass (before the main pass) when a caster is active.
    if (this.shadowCasterIndex >= 0) this.renderShadowPass(encoder);

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

    for (const mesh of this.opaque) this.drawMesh(pass, mesh);
    for (const mesh of this.transparent) this.drawMesh(pass, mesh);

    pass.end();

    // Resolve the HDR target through the post chain into the swap chain.
    if (this.postProcessing) {
      this.post.run(encoder, swapView, {
        fxaa: this.fxaa,
        bloom: this.bloom,
        bloomThreshold: this.bloomThreshold,
        bloomIntensity: this.bloomIntensity,
      });
    }

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Choose the shadow-casting directional light and fit an orthographic light
   * frustum to the opaque scene bounds, producing `_lightViewProj`. Sets
   * `shadowCasterIndex` (the light's index in the packed light array) or -1.
   */
  private prepareShadow(): void {
    this.shadowCasterIndex = -1;
    if (!this.shadows) return;
    this.createShadowResources(); // (re)allocate if size/enabled changed

    // Find the first directional light flagged castShadow, tracking its packed index.
    let caster: DirectionalLight | null = null;
    let packedIndex = -1;
    let i = 0;
    for (const light of this.lights) {
      if (light instanceof AmbientLight) continue;
      if (light instanceof DirectionalLight && light.castShadow) {
        caster = light;
        packedIndex = i;
        break;
      }
      i++;
    }
    if (!caster) return;

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
    if (minX > maxX) return; // nothing to shadow

    this._sceneCenter.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const radius = 0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);

    // Light direction: from the light position toward its target.
    caster.getWorldPosition(this._lightPos);
    caster.target.updateWorldMatrix(true, false);
    this._lightDir.set(
      caster.target.matrixWorld.elements[12] - this._lightPos.x,
      caster.target.matrixWorld.elements[13] - this._lightPos.y,
      caster.target.matrixWorld.elements[14] - this._lightPos.z,
    );
    if (this._lightDir.lengthSq() === 0) this._lightDir.set(0, -1, 0);
    this._lightDir.normalize();

    // Place the light camera back along -dir from the scene center.
    const eye = this._lightPos.set(
      this._sceneCenter.x - this._lightDir.x * radius,
      this._sceneCenter.y - this._lightDir.y * radius,
      this._sceneCenter.z - this._lightDir.z * radius,
    );
    const up = Math.abs(this._lightDir.y) > 0.99 ? UNIT_Z : UNIT_Y;
    // identity() first: lookAt only writes the upper 3x3.
    this._lightView.identity().lookAt(eye, this._sceneCenter, up).setPosition(eye).invert();

    // Fit the ortho box to the bounds transformed into light space.
    let lminX = Infinity, lminY = Infinity, lminZ = Infinity;
    let lmaxX = -Infinity, lmaxY = -Infinity, lmaxZ = -Infinity;
    for (let k = 0; k < 8; k++) {
      this._corner.set(k & 1 ? maxX : minX, k & 2 ? maxY : minY, k & 4 ? maxZ : minZ);
      this._corner.applyMatrix4(this._lightView);
      lminX = Math.min(lminX, this._corner.x); lmaxX = Math.max(lmaxX, this._corner.x);
      lminY = Math.min(lminY, this._corner.y); lmaxY = Math.max(lmaxY, this._corner.y);
      lminZ = Math.min(lminZ, this._corner.z); lmaxZ = Math.max(lmaxZ, this._corner.z);
    }
    // In view space the camera looks down -Z; depth distance d = -z.
    const near = -lmaxZ;
    const far = -lminZ;
    this._lightProj.makeOrthographic(lminX, lmaxX, lmaxY, lminY, near, far);
    this._lightViewProj.multiplyMatrices(this._lightProj, this._lightView);

    this.device.queue.writeBuffer(this.shadowLightBuffer, 0, new Float32Array(this._lightViewProj.elements));
    this.shadowCasterIndex = packedIndex;
  }

  /** Resolve `scene.environment` into the env bindings, rebuilding on change. */
  private prepareEnvironment(scene: Scene): void {
    const env = scene.environment;
    if (env && env.source) {
      const entry = this.textures.get(env);
      const key = `${env.id}:${env.version}`;
      if (key !== this.envKey) {
        this.envView = entry.view;
        this.envSampler = entry.sampler;
        this.envKey = key;
        this.buildFrameBindGroup();
      }
      this.envEnabled = true;
      this.envIntensity = scene.environmentIntensity;
      this.envMaxMip = Math.max(0, entry.texture.mipLevelCount - 1);
    } else {
      if (this.envKey !== '') {
        this.envView = this.textures.defaultWhiteView;
        this.envSampler = this.textures.defaultSampler;
        this.envKey = '';
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
      pass.setBindGroup(1, this.getMeshResources(mesh).bindGroup);
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
    f[63] = this.postProcessing ? 1 : 0;

    this.device.queue.writeBuffer(this.frameBuffer, 0, this.frameData);
    if (lightCount > 0) {
      this.device.queue.writeBuffer(this.lightBuffer, 0, this.lightData, 0, lightCount * (LIGHT_STRIDE / 4));
    }
  }

  exposure = 1.0;

  private drawMesh(pass: GPURenderPassEncoder, mesh: Mesh): void {
    const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    if (material instanceof LineBasicMaterial) {
      this.drawLine(pass, mesh, material);
      return;
    }
    if (!(material instanceof StandardMaterial)) return;
    const geometry = this.geometries.get(mesh.geometry);

    const instanced = mesh instanceof InstancedMesh;
    const skinned = !instanced && mesh instanceof SkinnedMesh && geometry.joints !== null && geometry.weights !== null;
    const morphed =
      !instanced && !skinned &&
      mesh.morphTargetInfluences.length > 0 &&
      !!mesh.geometry.morphAttributes.position?.length;
    const variant = instanced ? 'instanced' : skinned ? 'skinned' : morphed ? 'morph' : 'static';

    pass.setPipeline(this.pipelines.get(material, variant));

    // Group 1: model uniform (static/skinned/morph) or instance storage (instanced)
    if (instanced) {
      pass.setBindGroup(1, this.getInstancedResources(mesh as InstancedMesh).bindGroup);
    } else {
      pass.setBindGroup(1, this.getMeshResources(mesh).bindGroup);
    }
    pass.setBindGroup(2, this.getMaterialResources(material).bindGroup);

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
      pass.drawIndexed(geometry.drawCount, instanceCount);
    } else {
      pass.draw(geometry.drawCount, instanceCount);
    }
  }

  private drawLine(pass: GPURenderPassEncoder, mesh: Mesh, material: LineBasicMaterial): void {
    // The color stream is always present (white default), so lines need no setup.
    const geometry = this.geometries.get(mesh.geometry);

    pass.setPipeline(this.pipelines.getLine(material));
    pass.setBindGroup(1, this.getMeshResources(mesh).bindGroup);
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

  private getMeshResources(mesh: Mesh): MeshResources {
    let res = this.meshResources.get(mesh);
    if (!res) {
      const modelBuffer = this.device.createBuffer({
        size: MODEL_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.modelLayout,
        entries: [{ binding: 0, resource: { buffer: modelBuffer } }],
      });
      res = { modelBuffer, bindGroup };
      this.meshResources.set(mesh, res);
    }

    // update model + normal matrix every frame
    const data = new Float32Array(MODEL_SIZE / 4);
    data.set(mesh.matrixWorld.elements, 0);
    this._normalMatrix.getNormalMatrix(mesh.matrixWorld);
    const nm = this._normalMatrix.elements;
    // place 3x3 into mat4 columns
    data[16] = nm[0]; data[17] = nm[1]; data[18] = nm[2]; data[19] = 0;
    data[20] = nm[3]; data[21] = nm[4]; data[22] = nm[5]; data[23] = 0;
    data[24] = nm[6]; data[25] = nm[7]; data[26] = nm[8]; data[27] = 0;
    data[28] = 0; data[29] = 0; data[30] = 0; data[31] = 1;
    this.device.queue.writeBuffer(res.modelBuffer, 0, data);
    return res;
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
    return [m.map, m.normalMap, m.metalnessRoughnessMap, m.emissiveMap, m.occlusionMap]
      .map(id)
      .join('|');
  }
}
