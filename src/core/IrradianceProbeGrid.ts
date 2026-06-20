import { Object3D } from './Object3D';
import { Vector3 } from '../math/Vector3';

export interface IrradianceProbeGridOptions {
  dimensions?: [number, number, number];
  spacing?: Vector3;
}

/**
 * A world-axis-aligned regular grid of baked diffuse-irradiance probes represented
 * as SH-L2 coefficients. `position` is its minimum corner; rotation/scale are ignored.
 */
export class IrradianceProbeGrid extends Object3D {
  readonly isIrradianceProbeGrid = true;
  readonly dimensions: [number, number, number];
  readonly spacing: Vector3;
  /** Nine RGB SH coefficients per probe, x-fastest; null until baked or restored. */
  coefficients: Float32Array<ArrayBuffer> | null = null;
  version = 0;

  constructor(options: IrradianceProbeGridOptions = {}) {
    super();
    this.type = 'IrradianceProbeGrid';
    const dimensions = options.dimensions ?? [2, 2, 2];
    if (dimensions.some((n) => !Number.isInteger(n) || n < 1)) {
      throw new Error(`IrradianceProbeGrid: dimensions must be positive integers, got ${dimensions.join('x')}.`);
    }
    this.dimensions = [...dimensions];
    this.spacing = options.spacing?.clone() ?? new Vector3(2, 2, 2);
    if (this.spacing.x <= 0 || this.spacing.y <= 0 || this.spacing.z <= 0) {
      throw new Error('IrradianceProbeGrid: spacing components must be greater than zero.');
    }
  }

  get probeCount(): number {
    return this.dimensions[0] * this.dimensions[1] * this.dimensions[2];
  }

  setCoefficients(coefficients: Float32Array<ArrayBuffer>): this {
    const expected = this.probeCount * 9 * 4;
    if (coefficients.length !== expected) {
      throw new Error(`IrradianceProbeGrid: expected ${expected} coefficient values, got ${coefficients.length}.`);
    }
    this.coefficients = coefficients;
    this.version++;
    return this;
  }
}
