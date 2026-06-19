import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Color, type ColorInput } from '../math/Color';
import type { Box3 } from '../math/Box3';

// The 12 edges of a box as pairs of corner indices (corner bit n = axis n max).
const EDGES = [
  [0, 1], [1, 3], [3, 2], [2, 0], // z = min face
  [4, 5], [5, 7], [7, 6], [6, 4], // z = max face
  [0, 4], [1, 5], [2, 6], [3, 7], // connecting edges
];

/** A wireframe box outlining a {@link Box3}; call {@link update} to re-fit. */
export class Box3Helper extends LineSegments {
  constructor(box: Box3, color: ColorInput = new Color(1, 1, 0)) {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(EDGES.length * 2 * 3), 3));
    super(geometry, new LineBasicMaterial({ color }));
    this.type = 'Box3Helper';
    this.frustumCulled = false;
    this.update(box);
  }

  /** Rebuild the edge positions from `box` (in the box's own coordinate space). */
  update(box: Box3): void {
    const min = box.min;
    const max = box.max;
    // corner i: bit0 → x max, bit1 → y max, bit2 → z max
    const corner = (i: number): [number, number, number] => [
      (i & 1) ? max.x : min.x,
      (i & 2) ? max.y : min.y,
      (i & 4) ? max.z : min.z,
    ];

    const position = this.geometry.attributes.position;
    const arr = position.array as Float32Array;
    let o = 0;
    for (const [a, b] of EDGES) {
      const ca = corner(a);
      const cb = corner(b);
      arr[o++] = ca[0]; arr[o++] = ca[1]; arr[o++] = ca[2];
      arr[o++] = cb[0]; arr[o++] = cb[1]; arr[o++] = cb[2];
    }
    position.needsUpdate();
    this.geometry.version++;
  }
}
