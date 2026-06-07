import { Object3D } from './Object3D';
import { Color } from '../math/Color';

/** Root of a renderable scene graph, with optional background and ambient fill. */
export class Scene extends Object3D {
  readonly isScene = true;
  background: Color | null = null;
  /** Flat ambient irradiance applied to all PBR materials. */
  ambientColor = new Color(0, 0, 0);
  ambientIntensity = 0;

  constructor() {
    super();
    this.type = 'Scene';
  }
}
