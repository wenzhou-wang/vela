import { Object3D } from '../core/Object3D';
import { Color } from '../math/Color';

/** Base light: a colored, intensity-scaled emitter in the scene graph. */
export abstract class Light extends Object3D {
  readonly isLight = true;
  color = new Color(1, 1, 1);
  intensity: number;

  constructor(color: number | Color = 0xffffff, intensity = 1) {
    super();
    this.type = 'Light';
    if (typeof color === 'number') this.color.setHex(color);
    else this.color.copy(color);
    this.intensity = intensity;
  }
}
