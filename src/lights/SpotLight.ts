import { Light } from './Light';
import { Object3D } from '../core/Object3D';
import type { ColorInput } from '../math/Color';

/**
 * A cone-shaped light with perspective falloff, angular attenuation, and
 * optional shadow mapping.  Direction is from this object's world position
 * toward `target`.
 */
export class SpotLight extends Light {
  readonly isSpotLight = true;
  /** Half-angle of the light cone in radians (default = 60° full → PI/3 half). */
  angle: number;
  /** Fraction of `angle` used for the penumbra (soft edge). Range [0, 1]. */
  penumbra: number;
  /** Maximum illumination range; 0 = infinite. */
  distance: number;
  /** Falloff exponent (2 = physically correct inverse-square). */
  decay: number;
  /** When true and `renderer.shadows` is on, this light writes a shadow map tile. */
  castShadow = false;
  /** The spot aims from its world position toward this object. */
  readonly target: Object3D;

  constructor(
    color?: ColorInput,
    intensity = 1,
    distance = 0,
    angle = Math.PI / 3,
    penumbra = 0,
    decay = 2,
  ) {
    super(color, intensity);
    this.type = 'SpotLight';
    this.distance = distance;
    this.angle = angle;
    this.penumbra = penumbra;
    this.decay = decay;
    this.target = new Object3D();
  }
}
