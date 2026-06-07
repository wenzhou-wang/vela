import { Material } from './Material';
import { Color } from '../math/Color';

export interface LineBasicMaterialParams {
  color?: Color | number;
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
    if (params.color !== undefined) {
      if (typeof params.color === 'number') this.color.setHex(params.color);
      else this.color.copy(params.color);
    }
    if (params.vertexColors !== undefined) this.vertexColors = params.vertexColors;
    if (params.transparent !== undefined) this.transparent = params.transparent;
    if (params.opacity !== undefined) this.opacity = params.opacity;
    if (params.depthTest !== undefined) this.depthTest = params.depthTest;
    if (params.depthWrite !== undefined) this.depthWrite = params.depthWrite;
  }
}
