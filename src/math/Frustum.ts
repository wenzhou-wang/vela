import { Plane } from './Plane';
import type { Matrix4 } from './Matrix4';
import type { Sphere } from './Sphere';
import type { Box3 } from './Box3';
import { Vector3 } from './Vector3';

const _v = new Vector3();

/**
 * A view frustum as six inward-facing planes, extracted from a
 * view-projection matrix (Gribb-Hartmann) for WebGPU's [0, 1] clip depth.
 */
export class Frustum {
  readonly planes: [Plane, Plane, Plane, Plane, Plane, Plane];

  constructor() {
    this.planes = [new Plane(), new Plane(), new Plane(), new Plane(), new Plane(), new Plane()];
  }

  /**
   * Build from m = projection * view. Plane normals point inward.
   * Uses the [0, 1] depth convention: near = z-row, far = w-row - z-row.
   */
  setFromProjectionMatrix(m: Matrix4): this {
    const e = m.elements;
    // rows of the (column-major) matrix
    const xx = e[0], xy = e[4], xz = e[8], xw = e[12]; // row 0
    const yx = e[1], yy = e[5], yz = e[9], yw = e[13]; // row 1
    const zx = e[2], zy = e[6], zz = e[10], zw = e[14]; // row 2
    const wx = e[3], wy = e[7], wz = e[11], ww = e[15]; // row 3

    const p = this.planes;
    p[0].set(wx + xx, wy + xy, wz + xz, ww + xw).normalize(); // left
    p[1].set(wx - xx, wy - xy, wz - xz, ww - xw).normalize(); // right
    p[2].set(wx + yx, wy + yy, wz + yz, ww + yw).normalize(); // bottom
    p[3].set(wx - yx, wy - yy, wz - yz, ww - yw).normalize(); // top
    p[4].set(zx, zy, zz, zw).normalize(); // near ([0,1] depth)
    p[5].set(wx - zx, wy - zy, wz - zz, ww - zw).normalize(); // far
    return this;
  }

  intersectsSphere(sphere: Sphere): boolean {
    const { center, radius } = sphere;
    for (let i = 0; i < 6; i++) {
      if (this.planes[i].distanceToPoint(center) < -radius) return false;
    }
    return true;
  }

  intersectsBox(box: Box3): boolean {
    for (let i = 0; i < 6; i++) {
      const plane = this.planes[i];
      const n = plane.normal;
      // p-vertex: the box corner farthest along the plane normal
      _v.x = n.x > 0 ? box.max.x : box.min.x;
      _v.y = n.y > 0 ? box.max.y : box.min.y;
      _v.z = n.z > 0 ? box.max.z : box.min.z;
      if (plane.distanceToPoint(_v) < 0) return false;
    }
    return true;
  }
}
