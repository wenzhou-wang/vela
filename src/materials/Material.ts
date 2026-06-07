import { generateUUID } from '../math/MathUtils';

export type Side = 'front' | 'back' | 'double';
export type Blending = 'opaque' | 'blend';

/** Base material: render-state common to all shading models. */
export abstract class Material {
  readonly id: string = generateUUID();
  readonly isMaterial = true;
  name = '';
  abstract readonly type: string;

  side: Side = 'front';
  transparent = false;
  blending: Blending = 'opaque';
  opacity = 1;
  depthTest = true;
  depthWrite = true;
  /** Alpha test cutoff; 0 disables. */
  alphaTest = 0;

  /**
   * Identifies the pipeline variant. Materials with the same key share a
   * compiled pipeline. Recomputed by the renderer when shader features change.
   */
  needsPipelineRebuild = true;
}
