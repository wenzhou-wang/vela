import { Object3D } from './Object3D';
import { Color } from '../math/Color';
import type { Texture } from '../textures/Texture';
import type { Vector3 } from '../math/Vector3';

/** Procedural daylight sky (Preetham analytic model). */
export interface SkyOptions {
  /** Direction TOWARD the sun (world space; will be normalized). */
  sunDirection: Vector3;
  /** Atmospheric haze, 1.7 (crisp) .. 10 (hazy). Default 4. */
  turbidity?: number;
}

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
  /**
   * Procedural daylight sky. When set (and `environment` is null), the engine
   * generates an equirectangular sky texture on the GPU and uses it for IBL;
   * combine with `skybox = true` to also draw it as the background. Mutating
   * the options regenerates the sky automatically.
   */
  sky: SkyOptions | null = null;
  /** Draw the environment (or procedural sky) as the background. */
  skybox = false;
  /** Blur the skybox background: 0 = sharp, 1 = fully blurred (mip-based). */
  backgroundBlur = 0;

  constructor() {
    super();
    this.type = 'Scene';
  }
}
