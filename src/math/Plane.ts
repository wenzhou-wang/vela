import { Vector3 } from './Vector3';

/** A plane in Hessian normal form: normal·x + constant = 0. */
export class Plane {
  normal: Vector3;
  constant: number;

  constructor(normal = new Vector3(1, 0, 0), constant = 0) {
    this.normal = normal;
    this.constant = constant;
  }

  set(nx: number, ny: number, nz: number, constant: number): this {
    this.normal.set(nx, ny, nz);
    this.constant = constant;
    return this;
  }

  /** Scale so the normal is unit length (keeps distances metric). */
  normalize(): this {
    const inv = 1 / (this.normal.length() || 1);
    this.normal.multiplyScalar(inv);
    this.constant *= inv;
    return this;
  }

  distanceToPoint(point: Vector3): number {
    return this.normal.dot(point) + this.constant;
  }
}
