import { Object3D } from './Object3D';
import { Color } from '../math/Color';
import { Vector3 } from '../math/Vector3';

/**
 * Declarative GPU particle emitter config. All fields are plain values —
 * mutate them freely; the renderer re-reads them every frame (no flags).
 * `[start, end]` pairs interpolate linearly over each particle's life.
 */
export interface ParticleSystemOptions {
  /** Particle pool size (oldest particles recycle when full). Default 1000. */
  capacity?: number;
  /** Particles emitted per second. Default 100. */
  rate?: number;
  /** Particle lifetime in seconds, fixed or [min, max]. Default 2. */
  lifetime?: number | [number, number];
  /** Base emission velocity (world space). Default (0, 1, 0). */
  velocity?: Vector3;
  /** Random velocity jitter magnitude added per axis. Default 0.5. */
  spread?: number;
  /** Constant acceleration (world space). Default (0, 0, 0). */
  gravity?: Vector3;
  /** Quad size in world units, fixed or [start, end] over life. Default 0.1. */
  size?: number | [number, number];
  /** Tint, fixed or [start, end] over life. Default white. */
  color?: Color | [Color, Color];
  /** Opacity, fixed or [start, end] over life. Default [1, 0] (fade out). */
  opacity?: number | [number, number];
  /** 'additive' (default; glows, no sort artifacts) or 'alpha'. */
  blending?: 'additive' | 'alpha';
}

/**
 * A GPU-simulated particle emitter: a compute pass integrates a fixed-capacity
 * pool (positions/velocities/age live only on the GPU; the hot path allocates
 * nothing per particle), and one instanced draw renders camera-facing soft
 * discs. Emission happens at this object's world position — move it like any
 * Object3D to make trails.
 *
 * ```ts
 * const fire = new ParticleSystem({
 *   rate: 200,
 *   lifetime: [0.5, 1.2],
 *   velocity: new Vector3(0, 2, 0),
 *   spread: 0.4,
 *   size: [0.3, 0.05],
 *   color: [new Color(1, 0.6, 0.1), new Color(0.8, 0.1, 0.05)],
 * });
 * scene.add(fire);
 * ```
 */
export class ParticleSystem extends Object3D {
  readonly isParticleSystem = true;
  options: ParticleSystemOptions;
  /** Pause emission (existing particles keep simulating). */
  emitting = true;

  constructor(options: ParticleSystemOptions = {}) {
    super();
    this.type = 'ParticleSystem';
    const cap = options.capacity ?? 1000;
    if (!Number.isInteger(cap) || cap < 1 || cap > 1_000_000) {
      throw new Error(
        `ParticleSystem: capacity must be an integer in 1..1000000, got ${cap}. ` +
        `It is the GPU pool size; raise it if particles disappear too early.`,
      );
    }
    this.options = options;
  }
}
