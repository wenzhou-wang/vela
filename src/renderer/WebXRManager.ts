import type { Scene } from '../core/Scene';
import { PerspectiveCamera } from '../core/PerspectiveCamera';
import { RenderTarget } from '../core/RenderTarget';
import type { WebGPURenderer } from './WebGPURenderer';

interface XRViewLike {
  eye: string;
  projectionMatrix: Float32Array;
  transform: { matrix: Float32Array; inverse: { matrix: Float32Array } };
}
interface XRFrameLike {
  getViewerPose(space: unknown): { views: XRViewLike[] } | null;
}
interface XRSessionLike {
  requestReferenceSpace(type: string): Promise<unknown>;
  requestAnimationFrame(callback: (time: number, frame: XRFrameLike) => void): number;
  updateRenderState(state: { layers: unknown[]; depthNear?: number; depthFar?: number }): void;
  end(): Promise<void>;
}
interface XRSubImageLike {
  colorTexture: GPUTexture;
  viewport: { x: number; y: number; width: number; height: number };
  getViewDescriptor(): GPUTextureViewDescriptor;
}
interface XRBindingLike {
  createProjectionLayer(init: Record<string, unknown>): unknown;
  getViewSubImage(layer: unknown, view: XRViewLike): XRSubImageLike;
}

export interface WebXRStartOptions {
  mode?: 'immersive-vr' | 'immersive-ar';
  referenceSpace?: 'local' | 'local-floor' | 'bounded-floor' | 'unbounded';
  optionalFeatures?: string[];
  near?: number;
  far?: number;
  /** Called before each stereo frame, so application state/input can update once. */
  onFrame?: (time: number, frame: unknown) => void;
}

/** Experimental WebXR/WebGPU projection-layer integration (`XRGPUBinding`). */
export class WebXRManager {
  session: XRSessionLike | null = null;
  private binding: XRBindingLike | null = null;
  private layer: unknown = null;
  private space: unknown = null;
  private scene: Scene | null = null;
  private options: WebXRStartOptions = {};
  private running = false;
  private cameras = new Map<string, PerspectiveCamera>();
  private targets = new Map<string, RenderTarget>();

  constructor(private renderer: WebGPURenderer) {}

  static isSupported(): boolean {
    const nav = navigator as unknown as { xr?: unknown };
    return !!nav.xr && typeof (globalThis as { XRGPUBinding?: unknown }).XRGPUBinding === 'function';
  }

  async start(scene: Scene, options: WebXRStartOptions = {}): Promise<void> {
    if (!this.renderer.xrCompatible) {
      throw new Error('WebXRManager: create WebGPURenderer with { xrCompatible: true } before init().');
    }
    const nav = navigator as unknown as {
      xr?: { requestSession(mode: string, init: Record<string, unknown>): Promise<XRSessionLike> };
    };
    const Binding = (globalThis as unknown as {
      XRGPUBinding?: new(session: XRSessionLike, device: GPUDevice) => XRBindingLike;
    }).XRGPUBinding;
    if (!nav.xr || !Binding) throw new Error('WebXRManager: XRGPUBinding is unavailable in this browser.');
    const optional = options.optionalFeatures ?? [];
    this.session = await nav.xr.requestSession(options.mode ?? 'immersive-vr', {
      requiredFeatures: ['webgpu'], optionalFeatures: optional,
    });
    this.binding = new Binding(this.session, this.renderer.device);
    this.layer = this.binding.createProjectionLayer({
      colorFormat: 'rgba8unorm',
      textureUsage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    this.session.updateRenderState({ layers: [this.layer], depthNear: options.near ?? 0.1, depthFar: options.far ?? 1000 });
    this.space = await this.session.requestReferenceSpace(options.referenceSpace ?? 'local-floor');
    this.scene = scene;
    this.options = options;
    this.running = true;
    this.session.requestAnimationFrame(this.onFrame);
  }

  async end(): Promise<void> {
    this.running = false;
    await this.session?.end();
    this.session = null;
  }

  private onFrame = (time: number, frame: XRFrameLike): void => {
    if (!this.running || !this.session || !this.binding || !this.scene) return;
    this.options.onFrame?.(time, frame);
    const pose = frame.getViewerPose(this.space);
    if (pose) {
      for (const view of pose.views) this.renderView(view);
    }
    this.session.requestAnimationFrame(this.onFrame);
  };

  private renderView(view: XRViewLike): void {
    const sub = this.binding!.getViewSubImage(this.layer, view);
    const { width, height, x, y } = sub.viewport;
    let target = this.targets.get(view.eye);
    if (!target || target.width !== width || target.height !== height) {
      target = new RenderTarget(width, height);
      this.targets.set(view.eye, target);
    }
    let camera = this.cameras.get(view.eye);
    if (!camera) {
      camera = new PerspectiveCamera();
      camera.matrixAutoUpdate = false;
      this.cameras.set(view.eye, camera);
    }
    camera.near = this.options.near ?? 0.1;
    camera.far = this.options.far ?? 1000;
    camera.projectionMatrix.fromArray(view.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.matrixWorld.fromArray(view.transform.matrix);
    camera.matrixWorldInverse.fromArray(view.transform.inverse.matrix);
    this.renderer.render(this.scene!, camera, target);
    const descriptor = sub.getViewDescriptor();
    this.renderer.copyRenderTargetToTexture(target, sub.colorTexture, descriptor.baseArrayLayer ?? 0, x, y);
  }
}
