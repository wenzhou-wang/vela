import { Ray } from '../math/Ray';
import { Vector3 } from '../math/Vector3';
import { Sphere } from '../math/Sphere';
import { Object3D } from './Object3D';
import { Mesh } from './Mesh';
import type { Camera } from './Camera';

export interface Intersection {
  object: Mesh;
  /** Distance from the ray origin to the hit. */
  distance: number;
  point: Vector3;
}

const _sphere = new Sphere();

/**
 * Picks meshes along a ray, at the bounding-sphere level. Good for selection
 * and hover; precise per-triangle hits are a later refinement (see roadmap).
 */
export class Raycaster {
  ray = new Ray();
  near = 0;
  far = Infinity;

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

    _sphere.copy(geometry.boundingSphere).applyMatrix4(mesh.matrixWorld);
    const t = this.ray.intersectSphere(_sphere);
    if (t === null || t < this.near || t > this.far) return;

    hits.push({ object: mesh, distance: t, point: this.ray.at(t, new Vector3()) });
  }
}
