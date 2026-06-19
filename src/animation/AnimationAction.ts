import type { AnimationClip } from './AnimationClip';
import type { Object3D } from '../core/Object3D';

/**
 * One playing instance of an `AnimationClip` inside an `AnimationMixer`: its own
 * playback time, weight, and (optional) per-bone mask. The mixer blends all
 * enabled actions by weight each frame. Drive playback through the mixer; tweak
 * `weight` / `timeScale` / `loop` directly, or use `fadeTo` for ramps.
 */
export class AnimationAction {
  clip: AnimationClip;
  /** Blend weight; the mixer normalizes across enabled actions. */
  weight = 1;
  /** Playback time in seconds. */
  time = 0;
  /** Per-action speed multiplier (combined with the mixer's `timeScale`). */
  timeScale = 1;
  loop = true;
  paused = false;
  /** Disabled actions are skipped by the blend and pruned once faded to 0. */
  enabled = true;
  /**
   * Additive layer: instead of averaging into the base pose, add this clip's
   * delta from `referenceTime` on top of it (e.g. a recoil or breathing layer),
   * scaled by `weight`. Evaluated after the normal weighted blend.
   */
  additive = false;
  /** Pose subtracted to form the additive delta (the clip's rest frame). */
  referenceTime = 0;
  /**
   * When set, only tracks targeting a node in this set contribute — a per-bone
   * mask (e.g. an upper-body action layered over a full-body walk).
   */
  mask: Set<Object3D> | null = null;
  /**
   * Fired by the mixer when playback crosses one of the clip's `events`
   * (loop-aware). Hang gameplay (footsteps, hit-frames) off the timeline.
   */
  onEvent: ((name: string, time: number) => void) | null = null;

  // Linear weight ramp (crossfade): inactive when _fadeRate is 0.
  private _fadeTarget = 0;
  private _fadeRate = 0;

  constructor(clip: AnimationClip) {
    this.clip = clip;
  }

  /** Restrict this action to the given target nodes (per-bone mask). */
  setMask(targets: Object3D[] | null): this {
    this.mask = targets ? new Set(targets) : null;
    return this;
  }

  /**
   * Ramp `weight` toward `target` over `duration` seconds (0 = instant).
   * `enabled` is turned on so the ramp can run; reaching `target === 0`
   * disables the action so the mixer can prune it.
   */
  fadeTo(target: number, duration: number): this {
    this.enabled = true;
    if (duration <= 0) {
      this.weight = target;
      this._fadeRate = 0;
      if (target <= 0) this.enabled = false;
      return this;
    }
    this._fadeTarget = target;
    this._fadeRate = (target - this.weight) / duration;
    return this;
  }

  fadeIn(duration: number): this { return this.fadeTo(1, duration); }
  fadeOut(duration: number): this { return this.fadeTo(0, duration); }

  /** Advance any active weight ramp by `dt`; returns true while still fading. */
  advanceFade(dt: number): boolean {
    if (this._fadeRate === 0) return false;
    this.weight += this._fadeRate * dt;
    const reached = this._fadeRate > 0 ? this.weight >= this._fadeTarget : this.weight <= this._fadeTarget;
    if (reached) {
      this.weight = this._fadeTarget;
      this._fadeRate = 0;
      if (this._fadeTarget <= 0) this.enabled = false;
    }
    return this._fadeRate !== 0;
  }
}
