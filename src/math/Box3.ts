import { Vector3 } from './Vector3';
import type { Matrix4 } from './Matrix4';

const _points = [
  new Vector3(), new Vector3(), new Vector3(), new Vector3(),
  new Vector3(), new Vector3(), new Vector3(), new Vector3(),
];

/** An axis-aligned bounding box. */
export class Box3 {
  min: Vector3;
  max: Vector3;

  constructor(min = new Vector3(Infinity, Infinity, Infinity), max = new Vector3(-Infinity, -Infinity, -Infinity)) {
    this.min = min;
    this.max = max;
  }

  makeEmpty(): this {
    this.min.set(Infinity, Infinity, Infinity);
    this.max.set(-Infinity, -Infinity, -Infinity);
    return this;
  }

  isEmpty(): boolean {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }

  expandByPoint(p: Vector3): this {
    this.min.min(p);
    this.max.max(p);
    return this;
  }

  /** Expand from a flat positions array (xyz triples). */
  setFromBufferArray(array: ArrayLike<number>): this {
    this.makeEmpty();
    for (let i = 0; i < array.length; i += 3) {
      const x = array[i], y = array[i + 1], z = array[i + 2];
      if (x < this.min.x) this.min.x = x;
      if (y < this.min.y) this.min.y = y;
      if (z < this.min.z) this.min.z = z;
      if (x > this.max.x) this.max.x = x;
      if (y > this.max.y) this.max.y = y;
      if (z > this.max.z) this.max.z = z;
    }
    return this;
  }

  union(box: Box3): this {
    this.min.min(box.min);
    this.max.max(box.max);
    return this;
  }

  getCenter(target: Vector3): Vector3 {
    if (this.isEmpty()) return target.set(0, 0, 0);
    return target.addVectors(this.min, this.max).multiplyScalar(0.5);
  }

  getSize(target: Vector3): Vector3 {
    if (this.isEmpty()) return target.set(0, 0, 0);
    return target.subVectors(this.max, this.min);
  }

  /** Transform the 8 corners by a matrix and recompute the AABB. */
  applyMatrix4(m: Matrix4): this {
    if (this.isEmpty()) return this;
    _points[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(m);
    _points[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(m);
    _points[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(m);
    _points[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(m);
    _points[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(m);
    _points[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(m);
    _points[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(m);
    _points[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(m);
    this.makeEmpty();
    for (const p of _points) this.expandByPoint(p);
    return this;
  }

  copy(box: Box3): this {
    this.min.copy(box.min);
    this.max.copy(box.max);
    return this;
  }

  clone(): Box3 {
    return new Box3(this.min.clone(), this.max.clone());
  }
}
