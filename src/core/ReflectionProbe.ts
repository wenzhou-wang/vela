import { Object3D } from './Object3D';
import { PerspectiveCamera } from './PerspectiveCamera';
import { RenderTarget } from './RenderTarget';

export type ReflectionProbeRefresh = 'static' | 'every-n-frames';

export interface ReflectionProbeOptions {
  resolution?: number;
  near?: number;
  far?: number;
  radius?: number;
  intensity?: number;
  refresh?: ReflectionProbeRefresh;
  refreshInterval?: number;
}

/**
 * A local reflection capture. Add it to a Scene; the renderer refreshes it automatically.
 * The first four visible probes participate in nearest-two blending.
 */
export class ReflectionProbe extends Object3D {
  readonly isReflectionProbe = true;
  readonly resolution: number;
  readonly camera: PerspectiveCamera;
  readonly targets: readonly RenderTarget[];
  radius: number;
  intensity: number;
  refresh: ReflectionProbeRefresh;
  refreshInterval: number;
  /** Set true after moving a static probe to request another capture. */
  needsUpdate = true;

  constructor(options: ReflectionProbeOptions = {}) {
    super();
    this.type = 'ReflectionProbe';
    const resolution = options.resolution ?? 128;
    if (!Number.isInteger(resolution) || resolution < 1) {
      throw new Error(`ReflectionProbe: resolution must be a positive integer, got ${resolution}.`);
    }
    this.resolution = resolution;
    this.radius = options.radius ?? 10;
    this.intensity = options.intensity ?? 1;
    this.refresh = options.refresh ?? 'static';
    this.refreshInterval = Math.max(1, Math.floor(options.refreshInterval ?? 1));
    this.camera = new PerspectiveCamera(90, 1, options.near ?? 0.1, options.far ?? 1000);
    this.targets = Array.from({ length: 6 }, () => new RenderTarget(resolution, resolution));
  }
}
