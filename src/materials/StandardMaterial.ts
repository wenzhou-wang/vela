import { Material } from './Material';
import { Color } from '../math/Color';
import type { Texture } from '../textures/Texture';

export interface StandardMaterialParams {
  color?: Color | number;
  metalness?: number;
  roughness?: number;
  emissive?: Color | number;
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
  /** Index of refraction (KHR_materials_ior); default 1.5. */
  ior?: number;
  /** Dielectric specular strength [0,1] (KHR_materials_specular). */
  specularIntensity?: number;
  /** Dielectric specular tint (KHR_materials_specular). */
  specularColor?: Color | number;
  /** Sheen tint (KHR_materials_sheen); black (default) disables sheen. */
  sheenColor?: Color | number;
  /** Sheen roughness [0,1] (KHR_materials_sheen). */
  sheenRoughness?: number;
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

  map: Texture | null = null;
  normalMap: Texture | null = null;
  /** glTF packs metalness in B, roughness in G. */
  metalnessRoughnessMap: Texture | null = null;
  emissiveMap: Texture | null = null;
  occlusionMap: Texture | null = null;

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
    if (params.ior !== undefined) this.ior = params.ior;
    if (params.specularIntensity !== undefined) this.specularIntensity = params.specularIntensity;
    if (params.specularColor !== undefined) this.setColor(this.specularColor, params.specularColor);
    if (params.sheenColor !== undefined) this.setColor(this.sheenColor, params.sheenColor);
    if (params.sheenRoughness !== undefined) this.sheenRoughness = params.sheenRoughness;
  }

  private setColor(target: Color, value: Color | number): void {
    if (typeof value === 'number') target.setHex(value);
    else target.copy(value);
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
