import { Object3D } from './Object3D';
import { Color } from '../math/Color';
import type { Texture } from '../textures/Texture';
import type { Vector3 } from '../math/Vector3';

/**
 * Distance fog. Provide `density` for exponential-squared fog, or `near`/`far`
 * for linear fog (density takes precedence when both are set).
 */
export interface FogOptions {
  /** Fog color (linear); usually matched to the sky/background. */
  color: Color;
  /** Linear fog: distance where fog starts. Default 1. */
  near?: number;
  /** Linear fog: distance of full fog. Default 100. */
  far?: number;
  /** Exponential-squared fog density (e.g. 0.02). */
  density?: number;
  /** Fog thins with altitude when > 0 (e.g. 0.1); 0 = uniform. */
  heightFalloff?: number;
}

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
  /**
   * Distance fog applied to all lit materials (PBR and ShaderMaterial).
   * Debug helpers (lines) stay unfogged so they remain readable.
   */
  fog: FogOptions | null = null;
  /** Blur the skybox background: 0 = sharp, 1 = fully blurred (mip-based). */
  backgroundBlur = 0;

  constructor() {
    super();
    this.type = 'Scene';
  }
}
