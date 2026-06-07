import type { Object3D } from '../core/Object3D';
import { Quaternion } from '../math/Quaternion';

export type InterpolationMode = 'STEP' | 'LINEAR' | 'CUBICSPLINE';
export type TrackPath = 'translation' | 'rotation' | 'scale';

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
  /** Components per value: 3 for translation/scale, 4 for rotation. */
  readonly stride: number;

  private _cachedIndex = 0;

  constructor(
    target: Object3D,
    path: TrackPath,
    times: Float32Array,
    values: Float32Array,
    interpolation: InterpolationMode = 'LINEAR',
  ) {
    this.target = target;
    this.path = path;
    this.times = times;
    this.values = values;
    this.interpolation = interpolation;
    this.stride = path === 'rotation' ? 4 : 3;
  }

  get duration(): number {
    return this.times.length ? this.times[this.times.length - 1] : 0;
  }

  /** Evaluate at `time` and write the result into the target transform. */
  sample(time: number): void {
    const times = this.times;
    const n = times.length;
    if (n === 0) return;

    if (n === 1 || time <= times[0]) {
      this.write(0, 0, 0, 0);
      return;
    }
    if (time >= times[n - 1]) {
      this.write(n - 1, 0, 0, 0);
      return;
    }

    const i = this.findIndex(time);
    const t0 = times[i];
    const t1 = times[i + 1];
    const alpha = (time - t0) / (t1 - t0);

    if (this.interpolation === 'STEP') {
      this.write(i, 0, 0, 0);
    } else if (this.interpolation === 'CUBICSPLINE') {
      this.write(i, alpha, t1 - t0, i + 1);
    } else {
      this.writeLinear(i, i + 1, alpha);
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

  /** STEP / endpoint write: copy the value at key index `i`. */
  private write(i: number, _alpha: number, dt: number, nextIndex: number): void {
    const s = this.stride;
    const v = this.values;
    if (this.interpolation === 'CUBICSPLINE' && dt > 0) {
      this.writeCubic(i, nextIndex, _alpha, dt);
      return;
    }
    // CUBICSPLINE stores 3 vectors per key; the value is the middle one.
    const base = this.interpolation === 'CUBICSPLINE' ? i * s * 3 + s : i * s;
    this.applyVector(v, base);
  }

  private writeLinear(i0: number, i1: number, alpha: number): void {
    const s = this.stride;
    const v = this.values;
    const a = i0 * s;
    const b = i1 * s;
    const t = this.target;
    if (this.path === 'rotation') {
      _qa.set(v[a], v[a + 1], v[a + 2], v[a + 3]);
      _qb.set(v[b], v[b + 1], v[b + 2], v[b + 3]);
      _qa.slerp(_qb, alpha);
      t.quaternion.copy(_qa);
    } else {
      const dst = this.path === 'translation' ? t.position : t.scale;
      dst.x = v[a] + (v[b] - v[a]) * alpha;
      dst.y = v[a + 1] + (v[b + 1] - v[a + 1]) * alpha;
      dst.z = v[a + 2] + (v[b + 2] - v[a + 2]) * alpha;
    }
  }

  /** Cubic Hermite spline between keys i and i+1 (glTF CUBICSPLINE). */
  private writeCubic(i0: number, i1: number, t: number, dt: number): void {
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

    const out = _tmp;
    for (let k = 0; k < s; k++) {
      out[k] =
        h00 * v[p0 + k] +
        h10 * dt * v[m0 + k] +
        h01 * v[p1 + k] +
        h11 * dt * v[m1 + k];
    }

    const node = this.target;
    if (this.path === 'rotation') {
      _qa.set(out[0], out[1], out[2], out[3]).normalize();
      node.quaternion.copy(_qa);
    } else {
      const dst = this.path === 'translation' ? node.position : node.scale;
      dst.set(out[0], out[1], out[2]);
    }
  }

  private applyVector(v: Float32Array, base: number): void {
    const t = this.target;
    if (this.path === 'rotation') {
      t.quaternion.set(v[base], v[base + 1], v[base + 2], v[base + 3]);
    } else {
      const dst = this.path === 'translation' ? t.position : t.scale;
      dst.set(v[base], v[base + 1], v[base + 2]);
    }
  }
}

const _tmp = new Float32Array(4);
