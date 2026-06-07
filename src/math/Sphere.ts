import { Vector3 } from './Vector3';
import type { Matrix4 } from './Matrix4';

/** A bounding sphere. */
export class Sphere {
  center: Vector3;
  radius: number;

  constructor(center = new Vector3(), radius = -1) {
    this.center = center;
    this.radius = radius;
  }

  set(center: Vector3, radius: number): this {
    this.center.copy(center);
    this.radius = radius;
    return this;
  }

  copy(s: Sphere): this {
    this.center.copy(s.center);
    this.radius = s.radius;
    return this;
  }

  clone(): Sphere {
    return new Sphere(this.center.clone(), this.radius);
  }

  isEmpty(): boolean {
    return this.radius < 0;
  }

  /** Transform center by the matrix and scale radius by its max axis scale. */
  applyMatrix4(m: Matrix4): this {
    this.center.applyMatrix4(m);
    this.radius *= m.getMaxScaleOnAxis();
    return this;
  }
}
