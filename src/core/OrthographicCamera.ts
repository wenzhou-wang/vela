import { Camera } from './Camera';

/**
 * Orthographic projection camera (no perspective foreshortening). Useful for
 * CAD-style views, 2D overlays, and shadow-map light cameras.
 */
export class OrthographicCamera extends Camera {
  readonly isOrthographicCamera = true;

  left: number;
  right: number;
  top: number;
  bottom: number;
  near: number;
  far: number;
  /** Uniform scale on the frustum half-extents. */
  zoom = 1;

  constructor(left = -1, right = 1, top = 1, bottom = -1, near = 0.1, far = 2000) {
    super();
    this.type = 'OrthographicCamera';
    this.left = left;
    this.right = right;
    this.top = top;
    this.bottom = bottom;
    this.near = near;
    this.far = far;
    this.updateProjectionMatrix();
  }

  updateProjectionMatrix(): void {
    const dx = (this.right - this.left) / (2 * this.zoom);
    const dy = (this.top - this.bottom) / (2 * this.zoom);
    const cx = (this.right + this.left) / 2;
    const cy = (this.top + this.bottom) / 2;
    this.projectionMatrix.makeOrthographic(cx - dx, cx + dx, cy + dy, cy - dy, this.near, this.far);
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
}
