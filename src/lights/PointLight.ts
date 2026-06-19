import { Light } from './Light';
import type { ColorInput } from '../math/Color';

/** An omnidirectional light with physically-based inverse-square falloff. */
export class PointLight extends Light {
  readonly isPointLight = true;
  /** Maximum range of influence; 0 = infinite. */
  distance: number;
  /** Additional falloff exponent (2 = physically correct). */
  decay: number;
  /** Render this light's shadow into the cube-face shadow atlas (requires `renderer.shadows`). */
  castShadow = false;

  constructor(color?: ColorInput, intensity = 1, distance = 0, decay = 2) {
    super(color, intensity);
    this.type = 'PointLight';
    this.distance = distance;
    this.decay = decay;
  }
}
