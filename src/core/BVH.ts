import { Box3 } from '../math/Box3';
import { Vector3 } from '../math/Vector3';
import type { Ray } from '../math/Ray';
import type { BufferGeometry } from './BufferGeometry';

interface BVHNode {
  box: Box3;
  /** Leaf: index range into `triangles`. Internal nodes set count = 0. */
  start: number;
  count: number;
  left: BVHNode | null;
  right: BVHNode | null;
}

const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();

/**
 * A median-split bounding-volume hierarchy over a geometry's triangles, used to
 * skip the per-triangle linear scan when raycasting dense meshes. Built once and
 * cached on the geometry; traversal prunes subtrees whose AABB the ray misses.
 */
export class BVH {
  /** Triangle indices in leaf order (triangle `t` spans vertices 3t..3t+2). */
  readonly triangles: Uint32Array;
  private readonly root: BVHNode;

  private constructor(triangles: Uint32Array, root: BVHNode) {
    this.triangles = triangles;
    this.root = root;
  }

  /** Build a BVH from a geometry's position attribute (and optional index). */
  static build(geometry: BufferGeometry, leafSize = 4): BVH | null {
    const position = geometry.attributes.position;
    if (!position) return null;
    const index = geometry.index;
    const triCount = index ? Math.floor(index.count / 3) : Math.floor(position.count / 3);
    if (triCount === 0) return null;

    const tris = new Uint32Array(triCount);
    for (let t = 0; t < triCount; t++) tris[t] = t;

    // Precompute per-triangle bounds and centroids once.
    const boxes: Box3[] = new Array(triCount);
    const centroids = new Float32Array(triCount * 3);
    for (let t = 0; t < triCount; t++) {
      let ia: number, ib: number, ic: number;
      if (index) {
        ia = index.getX(t * 3); ib = index.getX(t * 3 + 1); ic = index.getX(t * 3 + 2);
      } else {
        ia = t * 3; ib = t * 3 + 1; ic = t * 3 + 2;
      }
      _a.set(position.getX(ia), position.getY(ia), position.getZ(ia));
      _b.set(position.getX(ib), position.getY(ib), position.getZ(ib));
      _c.set(position.getX(ic), position.getY(ic), position.getZ(ic));
      const box = new Box3().makeEmpty();
      box.expandByPoint(_a).expandByPoint(_b).expandByPoint(_c);
      boxes[t] = box;
      centroids[t * 3] = (_a.x + _b.x + _c.x) / 3;
      centroids[t * 3 + 1] = (_a.y + _b.y + _c.y) / 3;
      centroids[t * 3 + 2] = (_a.z + _b.z + _c.z) / 3;
    }

    const build = (start: number, count: number): BVHNode => {
      const box = new Box3().makeEmpty();
      for (let i = start; i < start + count; i++) box.union(boxes[tris[i]]);

      if (count <= leafSize) {
        return { box, start, count, left: null, right: null };
      }

      // Split on the axis with the widest spread of centroids.
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = start; i < start + count; i++) {
        const o = tris[i] * 3;
        minX = Math.min(minX, centroids[o]); maxX = Math.max(maxX, centroids[o]);
        minY = Math.min(minY, centroids[o + 1]); maxY = Math.max(maxY, centroids[o + 1]);
        minZ = Math.min(minZ, centroids[o + 2]); maxZ = Math.max(maxZ, centroids[o + 2]);
      }
      const ex = maxX - minX, ey = maxY - minY, ez = maxZ - minZ;
      const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;

      // Sort this slice by centroid on the chosen axis, then split at the median.
      const slice = Array.from(tris.subarray(start, start + count));
      slice.sort((p, q) => centroids[p * 3 + axis] - centroids[q * 3 + axis]);
      tris.set(slice, start);

      const mid = count >> 1;
      // Degenerate spread (all centroids equal): make a leaf to avoid recursion.
      if (ex === 0 && ey === 0 && ez === 0) {
        return { box, start, count, left: null, right: null };
      }
      const left = build(start, mid);
      const right = build(start + mid, count - mid);
      return { box, start, count: 0, left, right };
    };

    return new BVH(tris, build(0, triCount));
  }

  /**
   * Visit every triangle whose leaf AABB the ray may cross. `visit` receives the
   * triangle index; subtrees the ray misses are pruned. Order is not sorted —
   * the caller does the exact triangle test and sorts the resulting hits.
   */
  intersect(ray: Ray, visit: (triangleIndex: number) => void): void {
    const stack: BVHNode[] = [this.root];
    while (stack.length) {
      const node = stack.pop()!;
      if (ray.intersectBox(node.box) === null) continue;
      if (node.count > 0) {
        for (let i = node.start; i < node.start + node.count; i++) visit(this.triangles[i]);
      } else {
        if (node.left) stack.push(node.left);
        if (node.right) stack.push(node.right);
      }
    }
  }
}
