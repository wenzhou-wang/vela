import type { AnimationClip } from './AnimationClip';
import type { KeyframeTrack, TrackPath } from './KeyframeTrack';
import type { Object3D } from '../core/Object3D';
import { AnimationAction } from './AnimationAction';

/** One contributor (a clip's track + the action it belongs to) to a binding. */
interface BlendItem {
  track: KeyframeTrack;
  action: AnimationAction;
}

/** All tracks across all actions that drive the same (target node, path). */
interface Binding {
  target: Object3D;
  path: TrackPath;
  stride: number;
  items: BlendItem[];
}

/**
 * Plays and blends `AnimationClip`s. Each clip runs as an `AnimationAction`
 * with its own time/weight; the mixer samples every enabled action and blends
 * the results per (node, path) by weight — weighted average for translation/
 * scale/morph weights, normalized weighted sum (nlerp) for rotation. Bindings
 * are cached, so a steady action set costs no per-frame allocation.
 *
 * Single-clip use stays trivial: `new AnimationMixer().play(clip)`.
 */
export class AnimationMixer {
  readonly actions: AnimationAction[] = [];
  /** Global speed multiplier applied on top of each action's `timeScale`. */
  timeScale = 1;
  paused = false;
  /** Default loop mode applied to actions created by `play`. */
  loop = true;

  private bindings: Binding[] = [];
  private bindingsDirty = true;
  // Reusable scratch sized to the largest binding stride (grown on rebuild).
  private scratch = new Float32Array(4);
  private accum = new Float32Array(4);
  private ref = new Float32Array(4);

  /** Play `clip` exclusively at full weight (replaces any current actions). */
  play(clip: AnimationClip): this {
    this.actions.length = 0;
    const action = new AnimationAction(clip);
    action.loop = this.loop;
    this.actions.push(action);
    this.bindingsDirty = true;
    return this;
  }

  /** Add another clip as an action (for blending); returns it for tuning. */
  add(clip: AnimationClip, weight = 1): AnimationAction {
    const action = new AnimationAction(clip);
    action.weight = weight;
    action.loop = this.loop;
    this.actions.push(action);
    this.bindingsDirty = true;
    return action;
  }

  /**
   * Cross-fade to `clip` over `duration` seconds: the new action fades in from
   * 0→1 while every currently-enabled action fades out to 0 (and is pruned once
   * gone). Returns the new action.
   */
  crossFadeTo(clip: AnimationClip, duration: number): AnimationAction {
    for (const a of this.actions) if (a.enabled) a.fadeOut(duration);
    const next = new AnimationAction(clip);
    next.loop = this.loop;
    next.weight = 0;
    next.fadeIn(duration);
    this.actions.push(next);
    this.bindingsDirty = true;
    return next;
  }

  stop(): this {
    this.actions.length = 0;
    this.bindingsDirty = true;
    return this;
  }

  /** The primary (first) action's clip, or null — for single-clip callers. */
  get clip(): AnimationClip | null {
    return this.actions.length ? this.actions[0].clip : null;
  }

  /** The primary (first) action's time, or 0. */
  get time(): number {
    return this.actions.length ? this.actions[0].time : 0;
  }

  /** Jump every action to absolute time `t` (seconds) and re-blend. */
  setTime(t: number): this {
    for (const a of this.actions) a.time = t;
    this.evaluate();
    return this;
  }

  /** Advance all actions by `dt` seconds, then blend them into the targets. */
  update(dt: number): void {
    if (this.paused) return;

    let pruned = false;
    for (const a of this.actions) {
      if (!a.enabled) continue;
      a.advanceFade(dt);
      if (a.paused) continue;
      const d = a.clip.duration;
      const prev = a.time;
      const delta = dt * a.timeScale * this.timeScale;
      let next = prev + delta;
      if (d > 0) {
        if (a.loop) {
          next %= d;
          if (next < 0) next += d;
        } else {
          next = Math.max(0, Math.min(next, d));
        }
      }
      a.time = next;
      if (a.onEvent && a.clip.events.length > 0 && delta !== 0 && d > 0) {
        this.fireEvents(a, prev, prev + delta, d);
      }
    }

    // Prune actions that faded fully out, so the action set doesn't grow.
    for (let i = this.actions.length - 1; i >= 0; i--) {
      if (!this.actions[i].enabled && this.actions[i].weight <= 0) {
        this.actions.splice(i, 1);
        pruned = true;
      }
    }
    if (pruned) this.bindingsDirty = true;

    this.evaluate();
  }

  /**
   * Fire the clip's events crossed in (`prev`, `rawNext`] (loop-aware; `rawNext`
   * is the unwrapped target time). One fire per event per update — a `dt` larger
   * than the clip can skip repeats, not the events themselves.
   */
  private fireEvents(a: AnimationAction, prev: number, rawNext: number, d: number): void {
    const onEvent = a.onEvent!;
    const forward = rawNext >= prev;
    for (const ev of a.clip.events) {
      const e = ev.time;
      let hit: boolean;
      if (forward) {
        hit = (!a.loop || rawNext <= d) ? (e > prev && e <= rawNext) : (e > prev || e <= rawNext - d);
      } else {
        hit = (!a.loop || rawNext >= 0) ? (e <= prev && e > rawNext) : (e <= prev || e > rawNext + d);
      }
      if (hit) onEvent(ev.name, e);
    }
  }

  /** Sample every enabled action and blend the results into the target nodes. */
  private evaluate(): void {
    if (this.bindingsDirty) this.rebuildBindings();
    const { scratch, accum, ref } = this;

    for (const b of this.bindings) {
      const s = b.stride;
      let totalW = 0;
      let haveRef = false;
      for (let k = 0; k < s; k++) accum[k] = 0;

      for (const item of b.items) {
        const a = item.action;
        if (!a.enabled || a.weight <= 0) continue;
        item.track.evaluate(a.time, scratch);
        const w = a.weight;
        if (b.path === 'rotation') {
          // Align to the first contributor's hemisphere, then nlerp.
          if (!haveRef) { for (let k = 0; k < 4; k++) ref[k] = scratch[k]; haveRef = true; }
          const dot = scratch[0] * ref[0] + scratch[1] * ref[1] + scratch[2] * ref[2] + scratch[3] * ref[3];
          const sign = dot < 0 ? -w : w;
          for (let k = 0; k < 4; k++) accum[k] += sign * scratch[k];
        } else {
          for (let k = 0; k < s; k++) accum[k] += w * scratch[k];
        }
        totalW += w;
      }

      if (totalW <= 0) continue; // no active contributor: leave the node as-is
      this.write(b, accum, totalW);
    }
  }

  /** Write the blended `accum` (weighted by `totalW`) into the target. */
  private write(b: Binding, accum: Float32Array, totalW: number): void {
    const t = b.target;
    if (b.path === 'rotation') {
      const len = Math.hypot(accum[0], accum[1], accum[2], accum[3]);
      if (len > 0) {
        const inv = 1 / len;
        t.quaternion.set(accum[0] * inv, accum[1] * inv, accum[2] * inv, accum[3] * inv);
      }
      return;
    }
    const inv = 1 / totalW;
    if (b.path === 'weights') {
      const w = (t as Object3D & { morphTargetInfluences: number[] }).morphTargetInfluences;
      for (let k = 0; k < b.stride; k++) w[k] = accum[k] * inv;
    } else {
      const dst = b.path === 'translation' ? t.position : t.scale;
      dst.set(accum[0] * inv, accum[1] * inv, accum[2] * inv);
    }
  }

  /** Group every action's tracks by (target, path); size the scratch buffers. */
  private rebuildBindings(): void {
    const map = new Map<string, Binding>();
    let maxStride = 4;
    for (const action of this.actions) {
      for (const track of action.clip.tracks) {
        if (action.mask && !action.mask.has(track.target)) continue;
        const key = track.target.id + '|' + track.path;
        let binding = map.get(key);
        if (!binding) {
          binding = { target: track.target, path: track.path, stride: track.stride, items: [] };
          map.set(key, binding);
        }
        binding.items.push({ track, action });
        if (track.stride > maxStride) maxStride = track.stride;
      }
    }
    this.bindings = [...map.values()];
    if (maxStride > this.scratch.length) {
      this.scratch = new Float32Array(maxStride);
      this.accum = new Float32Array(maxStride);
    }
    this.bindingsDirty = false;
  }
}
