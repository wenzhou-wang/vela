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
  /** Draw `environment` as the background (requires `environment` to be set). */
  skybox = false;
  /** Blur the skybox background: 0 = sharp, 1 = fully blurred (mip-based). */
  backgroundBlur = 0;

  constructor() {
    super();
    this.type = 'Scene';
  }
}
