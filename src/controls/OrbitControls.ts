import type { PerspectiveCamera } from '../core/PerspectiveCamera';
import { Vector3 } from '../math/Vector3';
import { Spherical } from '../math/Spherical';
/** Mouse/touch orbit-pan-zoom camera controls with inertial damping. */
export class OrbitControls {
  enabled = true;
  target = new Vector3();

  minDistance = 0.01;
  maxDistance = Infinity;
  minPolarAngle = 0;
  maxPolarAngle = Math.PI;

  enableDamping = true;
  dampingFactor = 0.08;
  rotateSpeed = 1.0;
  zoomSpeed = 1.0;
  panSpeed = 1.0;
  autoRotate = false;
  autoRotateSpeed = 1.0;

  private camera: PerspectiveCamera;
  private domElement: HTMLElement;

  private spherical = new Spherical();
  private sphericalDelta = new Spherical(0, 0, 0);
  private panOffset = new Vector3();
  private scale = 1;

  private rotateStart = { x: 0, y: 0 };
  private panStart = { x: 0, y: 0 };
  private state: 'none' | 'rotate' | 'pan' = 'none';
  private pointers = new Map<number, { x: number; y: number }>();
  private prevPinch = 0;

  constructor(camera: PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.domElement = domElement;

    const offset = new Vector3().copy(camera.position).sub(this.target);
    this.spherical.setFromVector3(offset);

    this.bind();
  }

  private bind(): void {
    const el = this.domElement;
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', this.onPointerDown);
    el.addEventListener('wheel', this.onWheel, { passive: false });
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
  }

  dispose(): void {
    const el = this.domElement;
    el.removeEventListener('pointerdown', this.onPointerDown);
    el.removeEventListener('wheel', this.onWheel);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.enabled) return;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size === 1) {
      const panButton = e.button === 2 || e.button === 1 || e.shiftKey;
      this.state = panButton ? 'pan' : 'rotate';
      this.rotateStart.x = e.clientX;
      this.rotateStart.y = e.clientY;
      this.panStart.x = e.clientX;
      this.panStart.y = e.clientY;
    } else if (this.pointers.size === 2) {
      this.state = 'pan';
      this.prevPinch = this.pinchDistance();
    }
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.enabled || !this.pointers.has(e.pointerId)) return;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size === 2) {
      const dist = this.pinchDistance();
      if (this.prevPinch > 0) {
        if (dist > this.prevPinch) this.scale /= 1 + (dist / this.prevPinch - 1) * this.zoomSpeed;
        else this.scale *= 1 + (this.prevPinch / dist - 1) * this.zoomSpeed;
      }
      this.prevPinch = dist;
      return;
    }

    if (this.state === 'rotate') {
      const dx = e.clientX - this.rotateStart.x;
      const dy = e.clientY - this.rotateStart.y;
      const el = this.domElement;
      this.sphericalDelta.theta -= (2 * Math.PI * dx) / el.clientHeight * this.rotateSpeed;
      this.sphericalDelta.phi -= (2 * Math.PI * dy) / el.clientHeight * this.rotateSpeed;
      this.rotateStart.x = e.clientX;
      this.rotateStart.y = e.clientY;
    } else if (this.state === 'pan') {
      const dx = e.clientX - this.panStart.x;
      const dy = e.clientY - this.panStart.y;
      this.pan(dx, dy);
      this.panStart.x = e.clientX;
      this.panStart.y = e.clientY;
    }
  };

  private onPointerUp = (e: PointerEvent): void => {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) this.state = 'none';
    this.prevPinch = 0;
  };

  private onWheel = (e: WheelEvent): void => {
    if (!this.enabled) return;
    e.preventDefault();
    const factor = Math.pow(0.95, this.zoomSpeed);
    if (e.deltaY < 0) this.scale *= factor;
    else this.scale /= factor;
  };

  private pinchDistance(): number {
    const pts = [...this.pointers.values()];
    const dx = pts[0].x - pts[1].x;
    const dy = pts[0].y - pts[1].y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private _v = new Vector3();

  private pan(dx: number, dy: number): void {
    const el = this.domElement;
    const offset = this._v.copy(this.camera.position).sub(this.target);
    let targetDistance = offset.length();
    targetDistance *= Math.tan((this.camera.fov / 2) * Math.PI / 180);

    const te = this.camera.matrix.elements;
    // pan left (camera x axis)
    const panX = (2 * dx * targetDistance) / el.clientHeight * this.panSpeed;
    const panY = (2 * dy * targetDistance) / el.clientHeight * this.panSpeed;
    this.panOffset.x -= te[0] * panX;
    this.panOffset.y -= te[1] * panX;
    this.panOffset.z -= te[2] * panX;
    this.panOffset.x += te[4] * panY;
    this.panOffset.y += te[5] * panY;
    this.panOffset.z += te[6] * panY;
  }

  /** Frame an object/bounds: position the target at center and back off to fit. */
  setTarget(center: Vector3, distance?: number): void {
    this.target.copy(center);
    const offset = this._v.copy(this.camera.position).sub(this.target);
    this.spherical.setFromVector3(offset);
    if (distance !== undefined) this.spherical.radius = distance;
    this.update();
  }

  update(): boolean {
    const offset = this._v;
    if (this.autoRotate && this.state === 'none') {
      this.sphericalDelta.theta -= (2 * Math.PI / 60 / 60) * this.autoRotateSpeed * 60 * 0.016;
    }

    // With damping, apply only a fraction of the accumulated delta each frame and
    // let the remainder carry over (inertia). Applying the full delta every frame
    // would integrate one drag impulse to ~1/dampingFactor of its intended amount.
    const damp = this.enableDamping ? this.dampingFactor : 1;
    this.spherical.theta += this.sphericalDelta.theta * damp;
    this.spherical.phi += this.sphericalDelta.phi * damp;
    this.spherical.phi = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this.spherical.phi));
    this.spherical.makeSafe();

    this.spherical.radius *= this.scale;
    this.spherical.radius = Math.max(this.minDistance, Math.min(this.maxDistance, this.spherical.radius));

    this.target.addScaledVector(this.panOffset, damp);

    offset.setFromSpherical(this.spherical.radius, this.spherical.phi, this.spherical.theta);
    this.camera.position.copy(this.target).add(offset);
    this.camera.lookAt(this.target);

    if (this.enableDamping) {
      this.sphericalDelta.theta *= 1 - this.dampingFactor;
      this.sphericalDelta.phi *= 1 - this.dampingFactor;
      this.panOffset.multiplyScalar(1 - this.dampingFactor);
    } else {
      this.sphericalDelta.set(0, 0, 0);
      this.panOffset.set(0, 0, 0);
    }
    this.scale = 1;
    return true;
  }
}
