import { Light } from './Light';
import type { ColorInput } from '../math/Color';

/** Uniform ambient fill applied equally to all surfaces. */
export class AmbientLight extends Light {
  readonly isAmbientLight = true;

  constructor(color?: ColorInput, intensity = 1) {
    super(color, intensity);
    this.type = 'AmbientLight';
  }
}
