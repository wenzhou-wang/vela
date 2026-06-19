import { Material } from './Material';
import { Color, type ColorInput } from '../math/Color';

export interface LineBasicMaterialParams {
  color?: ColorInput;
  /** Multiply the base color by the geometry's per-vertex `color` attribute. */
  vertexColors?: boolean;
  transparent?: boolean;
  opacity?: number;
  depthTest?: boolean;
  depthWrite?: boolean;
}

/**
 * Unlit line material: a flat color (optionally modulated by per-vertex colors),
 * evaluated in the renderer's WGSL line shader. Used by the debug helpers.
 */
export class LineBasicMaterial extends Material {
  readonly type = 'LineBasicMaterial';

  color = new Color(1, 1, 1);
  vertexColors = false;

  constructor(params: LineBasicMaterialParams = {}) {
    super();
    if (params.color !== undefined) this.color.setFrom(params.color);
    if (params.vertexColors !== undefined) this.vertexColors = params.vertexColors;
    if (params.transparent !== undefined) this.transparent = params.transparent;
    if (params.opacity !== undefined) this.opacity = params.opacity;
    if (params.depthTest !== undefined) this.depthTest = params.depthTest;
    if (params.depthWrite !== undefined) this.depthWrite = params.depthWrite;
  }
}
