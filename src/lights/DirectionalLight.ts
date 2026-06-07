import { Light } from './Light';
import { Object3D } from '../core/Object3D';
import { Color } from '../math/Color';

/** A light infinitely far away, casting parallel rays toward its target. */
export class DirectionalLight extends Light {
  readonly isDirectionalLight = true;
  /** Direction is from this object's world position toward the target. */
  target = new Object3D();

  constructor(color: number | Color = 0xffffff, intensity = 1) {
    super(color, intensity);
    this.type = 'DirectionalLight';
    this.position.set(0, 1, 0);
  }
}
