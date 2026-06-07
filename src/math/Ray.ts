import { Vector3 } from './Vector3';
import type { Sphere } from './Sphere';
import type { Box3 } from './Box3';
import type { Matrix4 } from './Matrix4';

const _v = new Vector3();
const _edge1 = new Vector3();
const _edge2 = new Vector3();
const _pvec = new Vector3();
const _tvec = new Vector3();
const _qvec = new Vector3();

/** A ray with an origin and a normalized direction. */
export class Ray {
  origin: Vector3;
  direction: Vector3;

  constructor(origin = new Vector3(), direction = new Vector3(0, 0, -1)) {
    this.origin = origin;
    this.direction = direction;
  }

  set(origin: Vector3, direction: Vector3): this {
    this.origin.copy(origin);
    this.direction.copy(direction);
    return this;
  }

  copy(ray: Ray): this {
    this.origin.copy(ray.origin);
    this.direction.copy(ray.direction);
    return this;
  }

  at(t: number, target: Vector3): Vector3 {
    return target.copy(this.direction).multiplyScalar(t).add(this.origin);
  }

  /** Distance along the ray to the first sphere intersection, or null. */
  intersectSphere(sphere: Sphere): number | null {
    _v.subVectors(sphere.center, this.origin);
    const tca = _v.dot(this.direction);
    const d2 = _v.dot(_v) - tca * tca;
    const r2 = sphere.radius * sphere.radius;
    if (d2 > r2) return null;
    const thc = Math.sqrt(r2 - d2);
    const t0 = tca - thc;
    const t1 = tca + thc;
    if (t1 < 0) return null; // sphere behind the ray
    return t0 >= 0 ? t0 : t1; // origin outside vs inside
  }

  /** Distance along the ray to the first AABB intersection, or null. */
  intersectBox(box: Box3): number | null {
    const o = this.origin;
    const d = this.direction;
    const invX = 1 / d.x, invY = 1 / d.y, invZ = 1 / d.z;

    let tmin = ((invX >= 0 ? box.min.x : box.max.x) - o.x) * invX;
    let tmax = ((invX >= 0 ? box.max.x : box.min.x) - o.x) * invX;
    const tymin = ((invY >= 0 ? box.min.y : box.max.y) - o.y) * invY;
    const tymax = ((invY >= 0 ? box.max.y : box.min.y) - o.y) * invY;

    if (tmin > tymax || tymin > tmax) return null;
    if (tymin > tmin || Number.isNaN(tmin)) tmin = tymin;
    if (tymax < tmax || Number.isNaN(tmax)) tmax = tymax;

    const tzmin = ((invZ >= 0 ? box.min.z : box.max.z) - o.z) * invZ;
    const tzmax = ((invZ >= 0 ? box.max.z : box.min.z) - o.z) * invZ;

    if (tmin > tzmax || tzmin > tmax) return null;
    if (tzmin > tmin || Number.isNaN(tmin)) tmin = tzmin;
    if (tzmax < tmax || Number.isNaN(tmax)) tmax = tzmax;

    if (tmax < 0) return null;
    return tmin >= 0 ? tmin : tmax;
  }

  /**
   * Distance along the ray to a triangle intersection, or null. Möller–Trumbore.
   * When `backfaceCulling` is true, triangles facing away (CCW winding) are skipped.
   * If `target` is given, the hit point is written into it.
   */
  intersectTriangle(
    a: Vector3,
    b: Vector3,
    c: Vector3,
    backfaceCulling: boolean,
    target?: Vector3,
  ): number | null {
    _edge1.subVectors(b, a);
    _edge2.subVectors(c, a);
    _pvec.crossVectors(this.direction, _edge2);
    const det = _edge1.dot(_pvec);

    if (backfaceCulling) {
      if (det < 1e-12) return null; // back-facing or parallel
    } else if (det > -1e-12 && det < 1e-12) {
      return null; // ray parallel to triangle
    }

    const invDet = 1 / det;
    _tvec.subVectors(this.origin, a);
    const u = _tvec.dot(_pvec) * invDet;
    if (u < 0 || u > 1) return null;

    _qvec.crossVectors(_tvec, _edge1);
    const vbary = this.direction.dot(_qvec) * invDet;
    if (vbary < 0 || u + vbary > 1) return null;

    const t = _edge2.dot(_qvec) * invDet;
    if (t < 0) return null; // behind the ray origin
    if (target) this.at(t, target);
    return t;
  }

  /** Transform the ray by a matrix (origin as a point, direction as a vector). */
  applyMatrix4(m: Matrix4): this {
    this.origin.applyMatrix4(m);
    this.direction.transformDirection(m);
    return this;
  }
}
