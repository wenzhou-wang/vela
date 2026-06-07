import { Camera } from './Camera';
import { DEG2RAD } from '../math/MathUtils';

export class PerspectiveCamera extends Camera {
  readonly isPerspectiveCamera = true;

  /** Vertical field of view in degrees. */
  fov: number;
  aspect: number;
  near: number;
  far: number;

  constructor(fov = 50, aspect = 1, near = 0.1, far = 2000) {
    super();
    this.type = 'PerspectiveCamera';
    this.fov = fov;
    this.aspect = aspect;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    this.projectionMatrix.makePerspective(this.fov * DEG2RAD, this.aspect, this.near, this.far);
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
}
