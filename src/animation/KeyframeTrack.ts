import type { Object3D } from '../core/Object3D';
import { Quaternion } from '../math/Quaternion';

export type InterpolationMode = 'STEP' | 'LINEAR' | 'CUBICSPLINE';
export type TrackPath = 'translation' | 'rotation' | 'scale' | 'weights';

/** A node carrying animatable morph-target weights (a `Mesh`). */
interface MorphTarget extends Object3D {
  morphTargetInfluences: number[];
}

const _qa = new Quaternion();
const _qb = new Quaternion();

/**
 * A single animated channel: keyframe `times` driving one transform `path`
 * (translation/scale = vec3, rotation = quaternion) of a target node.
 *
 * For CUBICSPLINE the `values` array stores three vectors per key
 * (in-tangent, value, out-tangent), per the glTF 2.0 spec.
 */
export class KeyframeTrack {
  target: Object3D;
  path: TrackPath;
  times: Float32Array;
  values: Float32Array;
  interpolation: InterpolationMode;
  /**
   * Components per value: 3 for translation/scale, 4 for rotation, or the
   * morph-target count for weights (supplied via `valueSize`).
   */
  readonly stride: number;

  private _cachedIndex = 0;
  private _out: Float32Array;

  constructor(
    target: Object3D,
    path: TrackPath,
    times: Float32Array,
    values: Float32Array,
    interpolation: InterpolationMode = 'LINEAR',
    valueSize?: number,
  ) {
    this.target = target;
    this.path = path;
    this.times = times;
    this.values = values;
    this.interpolation = interpolation;
    this.stride = valueSize ?? (path === 'rotation' ? 4 : 3);
    this._out = new Float32Array(this.stride);
  }

  get duration(): number {
    return this.times.length ? this.times[this.times.length - 1] : 0;
  }

  /** Evaluate at `time` and write the result into the target transform. */
  sample(time: number): void {
    this.evaluate(time, this._out);
    this.apply(this._out);
  }

  /**
   * Evaluate at `time` into `out` (length ≥ `stride`) as raw components —
   * translation/scale = xyz, rotation = xyzw (unit), weights = per-target —
   * without touching the target node. The mixer uses this to blend clips.
   */
  evaluate(time: number, out: Float32Array): void {
    const times = this.times;
    const n = times.length;
    if (n === 0) return;

    if (n === 1 || time <= times[0]) {
      this.readKey(0, out);
      return;
    }
    if (time >= times[n - 1]) {
      this.readKey(n - 1, out);
      return;
    }

    const i = this.findIndex(time);
    const t0 = times[i];
    const t1 = times[i + 1];
    const alpha = (time - t0) / (t1 - t0);

    if (this.interpolation === 'STEP') {
      this.readKey(i, out);
    } else if (this.interpolation === 'CUBICSPLINE') {
      this.evalCubic(i, i + 1, alpha, t1 - t0, out);
    } else {
      this.evalLinear(i, i + 1, alpha, out);
    }
  }

  private findIndex(t: number): number {
    const times = this.times;
    // fast path: still in the cached interval?
    const c = this._cachedIndex;
    if (c < times.length - 1 && times[c] <= t && t < times[c + 1]) return c;

    let lo = 0;
    let hi = times.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) lo = mid + 1;
      else hi = mid - 1;
    }
    this._cachedIndex = hi;
    return hi;
  }

  /** STEP / endpoint: read the raw value at key index `i` into `out`. */
  private readKey(i: number, out: Float32Array): void {
    const s = this.stride;
    // CUBICSPLINE stores 3 vectors per key; the value is the middle one.
    const base = this.interpolation === 'CUBICSPLINE' ? i * s * 3 + s : i * s;
    for (let k = 0; k < s; k++) out[k] = this.values[base + k];
  }

  private evalLinear(i0: number, i1: number, alpha: number, out: Float32Array): void {
    const s = this.stride;
    const v = this.values;
    const a = i0 * s;
    const b = i1 * s;
    if (this.path === 'rotation') {
      _qa.set(v[a], v[a + 1], v[a + 2], v[a + 3]);
      _qb.set(v[b], v[b + 1], v[b + 2], v[b + 3]);
      _qa.slerp(_qb, alpha);
      out[0] = _qa.x; out[1] = _qa.y; out[2] = _qa.z; out[3] = _qa.w;
    } else {
      for (let k = 0; k < s; k++) out[k] = v[a + k] + (v[b + k] - v[a + k]) * alpha;
    }
  }

  /** Cubic Hermite spline between keys i0 and i1 (glTF CUBICSPLINE) into `out`. */
  private evalCubic(i0: number, i1: number, t: number, dt: number, out: Float32Array): void {
    const s = this.stride;
    const v = this.values;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    // per-key layout: [inTangent(s), value(s), outTangent(s)]
    const p0 = i0 * s * 3 + s;
    const m0 = i0 * s * 3 + 2 * s; // outTangent of key i0
    const p1 = i1 * s * 3 + s;
    const m1 = i1 * s * 3; // inTangent of key i1

    for (let k = 0; k < s; k++) {
      out[k] =
        h00 * v[p0 + k] +
        h10 * dt * v[m0 + k] +
        h01 * v[p1 + k] +
        h11 * dt * v[m1 + k];
    }

    if (this.path === 'rotation') {
      _qa.set(out[0], out[1], out[2], out[3]).normalize();
      out[0] = _qa.x; out[1] = _qa.y; out[2] = _qa.z; out[3] = _qa.w;
    }
  }

  /** Write raw components from `out` into the target transform. */
  private apply(out: Float32Array): void {
    const t = this.target;
    if (this.path === 'weights') {
      const w = (t as MorphTarget).morphTargetInfluences;
      for (let k = 0; k < this.stride; k++) w[k] = out[k];
    } else if (this.path === 'rotation') {
      t.quaternion.set(out[0], out[1], out[2], out[3]);
    } else {
      const dst = this.path === 'translation' ? t.position : t.scale;
      dst.set(out[0], out[1], out[2]);
    }
  }
}
