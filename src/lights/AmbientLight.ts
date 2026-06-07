import { Light } from './Light';
import { Color } from '../math/Color';

/** Uniform ambient fill applied equally to all surfaces. */
export class AmbientLight extends Light {
  readonly isAmbientLight = true;

  constructor(color: number | Color = 0xffffff, intensity = 1) {
    super(color, intensity);
    this.type = 'AmbientLight';
  }
}
