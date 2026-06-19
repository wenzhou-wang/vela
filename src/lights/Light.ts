import { Object3D } from '../core/Object3D';
import { Color, type ColorInput } from '../math/Color';

/** Base light: a colored, intensity-scaled emitter in the scene graph. */
export abstract class Light extends Object3D {
  readonly isLight = true;
  color = new Color(1, 1, 1);
  intensity: number;

  constructor(color?: ColorInput, intensity = 1) {
    super();
    this.type = 'Light';
    if (color !== undefined) this.color.setFrom(color);
    this.intensity = intensity;
  }
}
