import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';

/** Three lines from the origin: +X red, +Y green, +Z blue. */
export class AxesHelper extends LineSegments {
  constructor(size = 1) {
    // prettier-ignore
    const positions = new Float32Array([
      0, 0, 0, size, 0, 0,
      0, 0, 0, 0, size, 0,
      0, 0, 0, 0, 0, size,
    ]);
    // Vertex colors in linear space (R/G/B at full intensity).
    // prettier-ignore
    const colors = new Float32Array([
      1, 0, 0, 1,  1, 0, 0, 1,
      0, 1, 0, 1,  0, 1, 0, 1,
      0, 0, 1, 1,  0, 0, 1, 1,
    ]);

    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));
    geometry.setAttribute('color', new BufferAttribute(colors, 4));

    super(geometry, new LineBasicMaterial({ vertexColors: true }));
    this.type = 'AxesHelper';
    // Gizmos should not be hidden by the geometry they annotate.
    this.frustumCulled = false;
  }
}
