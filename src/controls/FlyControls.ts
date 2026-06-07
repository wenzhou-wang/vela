import type { Camera } from '../core/Camera';
import { Euler } from '../math/Euler';
import { Vector3 } from '../math/Vector3';

/**
 * First-person / fly camera controls. WASD strafes in the camera plane, Q/E (or
 * Space/Ctrl) move along the world up axis, and the mouse looks around — drag by
 * default, or grab the pointer with {@link lock} for a true free-look.
 *
 * Call {@link update} once per frame with the elapsed seconds so movement is
 * frame-rate independent.
 */
export class FlyControls {
  enabled = true;

  /** World units per second. */
  movementSpeed = 5;
  /** Radians of rotation per pixel of mouse motion. */
  lookSpeed = 0.0025;
  /** Multiplier applied while a boost key (Shift) is held. */
  boostFactor = 3;
  /** Clamp pitch to just under straight up/down to avoid gimbal flip. */
  maxPolarAngle = Math.PI / 2 - 0.01;
  /** When true, vertical motion follows world up; when false, the camera's own up. */
  verticalIsWorldUp = true;

  private camera: Camera;
  private domElement: HTMLElement;

  private yaw = 0;
  private pitch = 0;
  private readonly euler = new Euler(0, 0, 0, 'YXZ');

  private dragging = false;
  private locked = false;
  private readonly keys = new Set<string>();

  // scratch
  private readonly _forward = new Vector3();
  private readonly _right = new Vector3();
  private readonly _up = new Vector3();
  private readonly _move = new Vector3();
  private readonly _worldUp = new Vector3(0, 1, 0);

  constructor(camera: Camera, domElement: HTMLElement) {
    this.camera = camera;
    this.domElement = domElement;

    // Seed yaw/pitch from the camera's current facing direction (-Z).
    this._forward.set(0, 0, -1).applyQuaternion(camera.quaternion);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, this._forward.y)));
    this.yaw = Math.atan2(-this._forward.x, -this._forward.z);

    this.bind();
  }

  private bind(): void {
    const el = this.domElement;
    el.addEventListener('contextmenu', this.preventDefault);
    el.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    document.addEventListener('pointerlockchange', this.onLockChange);
  }

  dispose(): void {
    const el = this.domElement;
    el.removeEventListener('contextmenu', this.preventDefault);
    el.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    document.removeEventListener('pointerlockchange', this.onLockChange);
    this.keys.clear();
  }

  /** Request pointer lock for continuous free-look (no dragging needed). */
  lock(): void {
    this.domElement.requestPointerLock?.();
  }

  /** Release a pointer lock acquired via {@link lock}. */
  unlock(): void {
    if (this.locked) document.exitPointerLock?.();
  }

  private preventDefault = (e: Event): void => e.preventDefault();

  private onLockChange = (): void => {
    this.locked = document.pointerLockElement === this.domElement;
  };

  private onPointerDown = (e: PointerEvent): void => {
    if (!this.enabled) return;
    this.dragging = true;
    this.domElement.setPointerCapture?.(e.pointerId);
  };

  private onPointerUp = (e: PointerEvent): void => {
    this.dragging = false;
    this.domElement.releasePointerCapture?.(e.pointerId);
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.enabled || (!this.dragging && !this.locked)) return;
    this.yaw -= e.movementX * this.lookSpeed;
    this.pitch -= e.movementY * this.lookSpeed;
    this.pitch = Math.max(-this.maxPolarAngle, Math.min(this.maxPolarAngle, this.pitch));
  };

  private onKeyDown = (e: KeyboardEvent): void => {
    if (!this.enabled) return;
    this.keys.add(e.code);
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    this.keys.delete(e.code);
  };

  /** Integrate one frame of movement/rotation. `delta` is in seconds. */
  update(delta: number): void {
    if (!this.enabled) return;

    // Apply look first so movement uses the current facing.
    this.euler.set(this.pitch, this.yaw, 0, 'YXZ');
    this.camera.quaternion.setFromEuler(this.euler);

    const keys = this.keys;
    let fwd = 0, strafe = 0, rise = 0;
    if (keys.has('KeyW') || keys.has('ArrowUp')) fwd += 1;
    if (keys.has('KeyS') || keys.has('ArrowDown')) fwd -= 1;
    if (keys.has('KeyD') || keys.has('ArrowRight')) strafe += 1;
    if (keys.has('KeyA') || keys.has('ArrowLeft')) strafe -= 1;
    if (keys.has('KeyE') || keys.has('Space')) rise += 1;
    if (keys.has('KeyQ') || keys.has('ControlLeft')) rise -= 1;

    if (fwd === 0 && strafe === 0 && rise === 0) return;

    // Camera basis from its quaternion: -Z forward, +X right, +Y up.
    this._forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this._right.set(1, 0, 0).applyQuaternion(this.camera.quaternion);
    if (this.verticalIsWorldUp) this._up.copy(this._worldUp);
    else this._up.set(0, 1, 0).applyQuaternion(this.camera.quaternion);

    this._move
      .set(0, 0, 0)
      .addScaledVector(this._forward, fwd)
      .addScaledVector(this._right, strafe)
      .addScaledVector(this._up, rise);
    if (this._move.lengthSq() === 0) return;

    const speed = this.movementSpeed * (keys.has('ShiftLeft') || keys.has('ShiftRight') ? this.boostFactor : 1);
    this._move.normalize().multiplyScalar(speed * delta);
    this.camera.position.add(this._move);
  }
}
