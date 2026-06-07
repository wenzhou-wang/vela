import { Object3D } from './Object3D';
import { Color } from '../math/Color';
import type { Texture } from '../textures/Texture';

/** Root of a renderable scene graph, with optional background and ambient fill. */
export class Scene extends Object3D {
  readonly isScene = true;
  background: Color | null = null;
  /** Flat ambient irradiance applied to all PBR materials. */
  ambientColor = new Color(0, 0, 0);
  ambientIntensity = 0;
  /**
   * Equirectangular environment map for image-based lighting. When set, replaces
   * the flat ambient term with mip-prefiltered diffuse + specular indirect light.
   */
  environment: Texture | null = null;
  /** Multiplier on the environment's contribution. */
  environmentIntensity = 1;

  constructor() {
    super();
    this.type = 'Scene';
  }
}
