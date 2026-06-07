import { Ray } from '../math/Ray';
import { Vector2 } from '../math/Vector2';
import { Vector3 } from '../math/Vector3';
import { Sphere } from '../math/Sphere';
import { Matrix4 } from '../math/Matrix4';
import { Object3D } from './Object3D';
import { Mesh } from './Mesh';
import { BVH } from './BVH';
import type { BufferAttribute } from './BufferAttribute';
import type { BufferGeometry } from './BufferGeometry';
import type { Camera } from './Camera';

export interface Intersection {
  object: Mesh;
  /** Distance from the ray origin to the hit (world space). */
  distance: number;
  /** Hit point in world space. */
  point: Vector3;
  /** Triangle index (0-based) when a precise hit was found, else undefined. */
  faceIndex?: number;
  /** Barycentric-interpolated UV at the hit, when the geometry has UVs. */
  uv?: Vector2;
}

const _sphere = new Sphere();
const _inverseMatrix = new Matrix4();
const _localRay = new Ray();
const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _localPoint = new Vector3();
const _worldPoint = new Vector3();
const _uvA = new Vector2();
const _uvB = new Vector2();
const _uvC = new Vector2();
const _bary = new Vector3();

/**
 * Picks meshes along a ray. A bounding-sphere test rejects far-away meshes
 * cheaply; survivors are tested per-triangle in the mesh's local space, so hits
 * carry an exact point, the triangle index, and an interpolated UV.
 */
export class Raycaster {
  ray = new Ray();
  near = 0;
  far = Infinity;

  /**
   * When false, the per-triangle pass is skipped and meshes are picked at the
   * bounding-sphere level (faster, coarser). Defaults to true.
   */
  precise = true;

  /**
   * Triangle count at or above which a cached BVH accelerates the precise pass
   * instead of a linear scan. Set to `Infinity` to always scan linearly.
   */
  bvhThreshold = 256;

  /** Build the ray from normalized device coords (x, y in [-1, 1]). */
  setFromCamera(ndcX: number, ndcY: number, camera: Camera): this {
    camera.updateWorldMatrix(true, false);
    this.ray.origin.setFromMatrixPosition(camera.matrixWorld);
    // Unproject a point and aim the ray through it.
    this.ray.direction
      .set(ndcX, ndcY, 0.5)
      .applyMatrix4(camera.projectionMatrixInverse)
      .applyMatrix4(camera.matrixWorld)
      .sub(this.ray.origin)
      .normalize();
    return this;
  }

  /** Intersect a subtree; returns hits sorted nearest-first. */
  intersect(root: Object3D, recursive = true): Intersection[] {
    const hits: Intersection[] = [];
    root.updateWorldMatrix(true, recursive);
    const visit = (object: Object3D) => {
      if (object instanceof Mesh && object.visible) this.testMesh(object, hits);
      if (recursive) for (const child of object.children) visit(child);
    };
    visit(root);
    hits.sort((a, b) => a.distance - b.distance);
    return hits;
  }

  private testMesh(mesh: Mesh, hits: Intersection[]): void {
    const geometry = mesh.geometry;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    if (!geometry.boundingSphere || geometry.boundingSphere.isEmpty()) return;

    // Broad phase: reject meshes whose world bounding sphere the ray misses.
    _sphere.copy(geometry.boundingSphere).applyMatrix4(mesh.matrixWorld);
    const sphereT = this.ray.intersectSphere(_sphere);
    if (sphereT === null) return;

    if (!this.precise || !geometry.attributes.position) {
      if (sphereT < this.near || sphereT > this.far) return;
      hits.push({ object: mesh, distance: sphereT, point: this.ray.at(sphereT, new Vector3()) });
      return;
    }

    // Narrow phase: test triangles in local space, where the geometry lives.
    _inverseMatrix.copy(mesh.matrixWorld).invert();
    _localRay.copy(this.ray).applyMatrix4(_inverseMatrix);
    this.testTriangles(mesh, hits);
  }

  private testTriangles(mesh: Mesh, hits: Intersection[]): void {
    const geometry = mesh.geometry;
    const position = geometry.attributes.position;
    const index = geometry.index;
    const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    // Double-sided materials are pickable from behind.
    const culling = material ? material.side !== 'double' : true;

    const triCount = index ? index.count / 3 : position.count / 3;
    if (triCount >= this.bvhThreshold) {
      const bvh = getBVH(geometry);
      if (bvh) {
        bvh.intersect(_localRay, (t) => this.testTriangle(mesh, t, culling, hits));
        return;
      }
    }
    for (let t = 0; t < triCount; t++) this.testTriangle(mesh, t, culling, hits);
  }

  /** Exact Möller–Trumbore test of triangle `t` (local space), pushing any hit. */
  private testTriangle(mesh: Mesh, t: number, culling: boolean, hits: Intersection[]): void {
    const geometry = mesh.geometry;
    const position = geometry.attributes.position;
    const uv = geometry.attributes.uv;
    const index = geometry.index;

    let ia: number, ib: number, ic: number;
    if (index) {
      ia = index.getX(t * 3);
      ib = index.getX(t * 3 + 1);
      ic = index.getX(t * 3 + 2);
    } else {
      ia = t * 3;
      ib = t * 3 + 1;
      ic = t * 3 + 2;
    }
    fromAttr(_a, position, ia);
    fromAttr(_b, position, ib);
    fromAttr(_c, position, ic);

    const localT = _localRay.intersectTriangle(_a, _b, _c, culling, _localPoint);
    if (localT === null) return;

    // Convert the local hit point back to world space for a true distance.
    _worldPoint.copy(_localPoint).applyMatrix4(mesh.matrixWorld);
    const distance = _worldPoint.distanceTo(this.ray.origin);
    if (distance < this.near || distance > this.far) return;

    const hit: Intersection = {
      object: mesh,
      distance,
      point: _worldPoint.clone(),
      faceIndex: t,
    };

    if (uv) {
      barycentric(_localPoint, _a, _b, _c, _bary);
      _uvA.set(uv.getX(ia), uv.getY(ia));
      _uvB.set(uv.getX(ib), uv.getY(ib));
      _uvC.set(uv.getX(ic), uv.getY(ic));
      hit.uv = new Vector2(
        _uvA.x * _bary.x + _uvB.x * _bary.y + _uvC.x * _bary.z,
        _uvA.y * _bary.x + _uvB.y * _bary.y + _uvC.y * _bary.z,
      );
    }

    hits.push(hit);
  }
}

function fromAttr(target: Vector3, attr: BufferAttribute, i: number): Vector3 {
  return target.set(attr.getX(i), attr.getY(i), attr.getZ(i));
}

/** BVH cache keyed by geometry; rebuilt when the geometry's `version` changes. */
const _bvhCache = new WeakMap<BufferGeometry, { version: number; bvh: BVH | null }>();

function getBVH(geometry: BufferGeometry): BVH | null {
  let entry = _bvhCache.get(geometry);
  if (!entry || entry.version !== geometry.version) {
    entry = { version: geometry.version, bvh: BVH.build(geometry) };
    _bvhCache.set(geometry, entry);
  }
  return entry.bvh;
}

const _v0 = new Vector3();
const _v1 = new Vector3();
const _v2 = new Vector3();

/** Barycentric coordinates of point p within triangle (a, b, c). */
function barycentric(p: Vector3, a: Vector3, b: Vector3, c: Vector3, target: Vector3): Vector3 {
  _v0.subVectors(b, a);
  _v1.subVectors(c, a);
  _v2.subVectors(p, a);
  const d00 = _v0.dot(_v0);
  const d01 = _v0.dot(_v1);
  const d11 = _v1.dot(_v1);
  const d20 = _v2.dot(_v0);
  const d21 = _v2.dot(_v1);
  const denom = d00 * d11 - d01 * d01 || 1;
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  return target.set(1 - v - w, v, w);
}
