import { Material } from './Material';
import { Color, type ColorInput } from '../math/Color';
import type { Texture } from '../textures/Texture';

export interface StandardMaterialParams {
  color?: ColorInput;
  metalness?: number;
  roughness?: number;
  emissive?: ColorInput;
  emissiveIntensity?: number;
  map?: Texture | null;
  normalMap?: Texture | null;
  normalScale?: number;
  metalnessRoughnessMap?: Texture | null;
  emissiveMap?: Texture | null;
  occlusionMap?: Texture | null;
  occlusionStrength?: number;
  side?: 'front' | 'back' | 'double';
  transparent?: boolean;
  opacity?: number;
  alphaTest?: number;
  /** Clear-coat layer strength [0,1] (KHR_materials_clearcoat). */
  clearcoat?: number;
  /** Clear-coat layer roughness [0,1]. */
  clearcoatRoughness?: number;
  /** Per-texel clearcoat factor (R channel, multiplied by clearcoat factor). */
  clearcoatMap?: Texture | null;
  /** Per-texel clearcoat roughness (G channel, multiplied by clearcoatRoughness). */
  clearcoatRoughnessMap?: Texture | null;
  /** Index of refraction (KHR_materials_ior); default 1.5. */
  ior?: number;
  /** Dielectric specular strength [0,1] (KHR_materials_specular). */
  specularIntensity?: number;
  /** Dielectric specular tint (KHR_materials_specular). */
  specularColor?: ColorInput;
  /** Sheen tint (KHR_materials_sheen); black (default) disables sheen. */
  sheenColor?: ColorInput;
  /** Sheen roughness [0,1] (KHR_materials_sheen). */
  sheenRoughness?: number;
  /** Transmission factor [0,1] (KHR_materials_transmission). */
  transmission?: number;
  /** Volume thickness (KHR_materials_volume). */
  thickness?: number;
  /** Attenuation distance (KHR_materials_volume); 0 = none. */
  attenuationDistance?: number;
  /** Attenuation color (KHR_materials_volume). */
  attenuationColor?: ColorInput;
  /** Linear height field sampled from R for parallax occlusion mapping. */
  heightMap?: Texture | null;
  parallaxScale?: number;
  parallaxMinLayers?: number;
  parallaxMaxLayers?: number;
}

/**
 * Physically based metallic-roughness material (glTF 2.0 compatible).
 * Cook-Torrance GGX specular + Lambertian diffuse, evaluated in the renderer's
 * WGSL PBR shader.
 */
export class StandardMaterial extends Material {
  readonly type = 'StandardMaterial';

  color = new Color(1, 1, 1);
  metalness = 1.0;
  roughness = 1.0;
  emissive = new Color(0, 0, 0);
  emissiveIntensity = 1.0;
  normalScale = 1.0;
  occlusionStrength = 1.0;
  /** Clear-coat layer (KHR_materials_clearcoat): a thin dielectric specular coat. */
  clearcoat = 0.0;
  clearcoatRoughness = 0.0;
  /** Index of refraction (KHR_materials_ior); 1.5 → the default 0.04 dielectric F0. */
  ior = 1.5;
  /** Dielectric specular strength and tint (KHR_materials_specular). */
  specularIntensity = 1.0;
  specularColor = new Color(1, 1, 1);
  /** Sheen layer (KHR_materials_sheen): a cloth-like retroreflective lobe. */
  sheenColor = new Color(0, 0, 0);
  sheenRoughness = 0.0;
  /** Transmission factor (KHR_materials_transmission): glass-like refraction. */
  transmission = 0.0;
  /** Volume thickness (KHR_materials_volume); 0 = thin-walled. */
  thickness = 0.0;
  /** Distance light travels before attenuating to `attenuationColor`; 0 = none. */
  attenuationDistance = 0.0;
  attenuationColor = new Color(1, 1, 1);
  parallaxScale = 0.0;
  parallaxMinLayers = 8;
  parallaxMaxLayers = 24;

  map: Texture | null = null;
  normalMap: Texture | null = null;
  /** glTF packs metalness in B, roughness in G. */
  metalnessRoughnessMap: Texture | null = null;
  emissiveMap: Texture | null = null;
  occlusionMap: Texture | null = null;
  /** Per-texel clearcoat factor (R channel). */
  clearcoatMap: Texture | null = null;
  /** Per-texel clearcoat roughness (G channel). */
  clearcoatRoughnessMap: Texture | null = null;
  heightMap: Texture | null = null;

  constructor(params: StandardMaterialParams = {}) {
    super();
    if (params.color !== undefined) this.setColor(this.color, params.color);
    if (params.emissive !== undefined) this.setColor(this.emissive, params.emissive);
    if (params.metalness !== undefined) this.metalness = params.metalness;
    if (params.roughness !== undefined) this.roughness = params.roughness;
    if (params.emissiveIntensity !== undefined) this.emissiveIntensity = params.emissiveIntensity;
    if (params.normalScale !== undefined) this.normalScale = params.normalScale;
    if (params.occlusionStrength !== undefined) this.occlusionStrength = params.occlusionStrength;
    if (params.map !== undefined) this.map = params.map;
    if (params.normalMap !== undefined) this.normalMap = params.normalMap;
    if (params.metalnessRoughnessMap !== undefined) this.metalnessRoughnessMap = params.metalnessRoughnessMap;
    if (params.emissiveMap !== undefined) this.emissiveMap = params.emissiveMap;
    if (params.occlusionMap !== undefined) this.occlusionMap = params.occlusionMap;
    if (params.side !== undefined) this.side = params.side;
    if (params.transparent !== undefined) this.transparent = params.transparent;
    if (params.opacity !== undefined) this.opacity = params.opacity;
    if (params.alphaTest !== undefined) this.alphaTest = params.alphaTest;
    if (params.clearcoat !== undefined) this.clearcoat = params.clearcoat;
    if (params.clearcoatRoughness !== undefined) this.clearcoatRoughness = params.clearcoatRoughness;
    if (params.clearcoatMap !== undefined) this.clearcoatMap = params.clearcoatMap ?? null;
    if (params.clearcoatRoughnessMap !== undefined) this.clearcoatRoughnessMap = params.clearcoatRoughnessMap ?? null;
    if (params.ior !== undefined) this.ior = params.ior;
    if (params.specularIntensity !== undefined) this.specularIntensity = params.specularIntensity;
    if (params.specularColor !== undefined) this.setColor(this.specularColor, params.specularColor);
    if (params.sheenColor !== undefined) this.setColor(this.sheenColor, params.sheenColor);
    if (params.sheenRoughness !== undefined) this.sheenRoughness = params.sheenRoughness;
    if (params.transmission !== undefined) this.transmission = params.transmission;
    if (params.thickness !== undefined) this.thickness = params.thickness;
    if (params.attenuationDistance !== undefined) this.attenuationDistance = params.attenuationDistance;
    if (params.attenuationColor !== undefined) this.setColor(this.attenuationColor, params.attenuationColor);
    if (params.heightMap !== undefined) this.heightMap = params.heightMap;
    if (params.parallaxScale !== undefined) this.parallaxScale = params.parallaxScale;
    if (params.parallaxMinLayers !== undefined) this.parallaxMinLayers = params.parallaxMinLayers;
    if (params.parallaxMaxLayers !== undefined) this.parallaxMaxLayers = params.parallaxMaxLayers;
  }

  private setColor(target: Color, value: ColorInput): void {
    target.setFrom(value);
  }

  /** Bitmask describing which texture maps are present (drives shader variants). */
  getFeatureKey(): number {
    let key = 0;
    if (this.map) key |= 1;
    if (this.normalMap) key |= 2;
    if (this.metalnessRoughnessMap) key |= 4;
    if (this.emissiveMap) key |= 8;
    if (this.occlusionMap) key |= 16;
    if (this.alphaTest > 0) key |= 32;
    if (this.side === 'double') key |= 64;
    if (this.transparent) key |= 128;
    return key;
  }
}
