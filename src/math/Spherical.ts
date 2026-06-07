import type { Vector3 } from './Vector3';

const EPS = 1e-6;

/** Spherical coordinates: radius, polar angle (phi from +Y), azimuth (theta). */
export class Spherical {
  radius: number;
  phi: number;
  theta: number;

  constructor(radius = 1, phi = 0, theta = 0) {
    this.radius = radius;
    this.phi = phi;
    this.theta = theta;
  }

  set(radius: number, phi: number, theta: number): this {
    this.radius = radius;
    this.phi = phi;
    this.theta = theta;
    return this;
  }

  setFromVector3(v: Vector3): this {
    this.radius = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (this.radius === 0) {
      this.theta = 0;
      this.phi = 0;
    } else {
      this.theta = Math.atan2(v.x, v.z);
      this.phi = Math.acos(Math.min(1, Math.max(-1, v.y / this.radius)));
    }
    return this;
  }

  /** Clamp phi to avoid the poles (gimbal flip). */
  makeSafe(): this {
    this.phi = Math.max(EPS, Math.min(Math.PI - EPS, this.phi));
    return this;
  }
}
