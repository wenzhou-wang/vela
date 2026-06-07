import { Mesh } from './Mesh';
import type { BufferGeometry } from './BufferGeometry';
import type { LineBasicMaterial } from '../materials/LineBasicMaterial';

/**
 * A set of disconnected line segments: the position attribute is read in pairs
 * (0-1, 2-3, …) and drawn with `line-list` topology. Reuses the scene-graph and
 * culling machinery of {@link Mesh}; the renderer routes it to the line pipeline.
 */
export class LineSegments extends Mesh {
  readonly isLineSegments = true;

  constructor(geometry: BufferGeometry, material: LineBasicMaterial) {
    super(geometry, material);
    this.type = 'LineSegments';
  }
}
