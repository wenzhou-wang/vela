import { Mesh } from './Mesh';
import { BufferGeometry } from './BufferGeometry';
import { BufferAttribute } from './BufferAttribute';
import { StandardMaterial } from '../materials/StandardMaterial';
import { Vector3 } from '../math/Vector3';
import type { Material } from '../materials/Material';
import type { Object3D } from './Object3D';

export interface TrailOptions {
  /** Surface material (default: an emissive double-sided StandardMaterial). */
  material?: Material;
  /** Position-history capacity (default 32). */
  maxPoints?: number;
  /** Ribbon half-width source: width at the newest point (default 0.2). */
  width?: number;
  /** Width at the oldest point (default 0 → taper to a point). */
  widthTail?: number;
  /** Emit a new point only after the emitter moves at least this far (default 0.05). */
  minDistance?: number;
}

const _p = new Vector3(), _q = new Vector3(), _dir = new Vector3();
const _side = new Vector3(), _view = new Vector3(), _up = new Vector3();

/**
 * A ribbon that trails an object's motion: each `update(cameraPosition)` samples
 * `target`'s world position into a fixed-capacity history and rebuilds a
 * camera-facing triangle strip through it (world space — keep the trail itself at
 * the scene root). Reuses the standard mesh path, so any `Material` works; drive
 * it from your loop like `controls.update()`. Geometry is offline-verifiable.
 */
export class TrailRenderer extends Mesh {
  readonly isTrail = true;
  /** Emitter sampled each update. Defaults to the trail itself — set it to your moving object. */
  target: Object3D;
  maxPoints: number;
  width: number;
  widthTail: number;
  minDistance: number;

  private points: Float32Array; // oldest → newest, 3 floats each
  private count = 0;

  constructor(options: TrailOptions = {}) {
    const maxPoints = Math.max(2, options.maxPoints ?? 32);
    const verts = maxPoints * 2;
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(verts * 3), 3));
    geometry.setAttribute('normal', new BufferAttribute(new Float32Array(verts * 3), 3));
    geometry.setAttribute('uv', new BufferAttribute(new Float32Array(verts * 2), 2));
    // Static index over (maxPoints-1) quads; unused quads stay degenerate.
    const index = new Uint32Array((maxPoints - 1) * 6);
    for (let i = 0; i < maxPoints - 1; i++) {
      const a = i * 2;
      index.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], i * 6);
    }
    geometry.setIndex(new BufferAttribute(index, 1));

    super(geometry, options.material ?? new StandardMaterial({
      color: [1, 1, 1], emissive: [0.5, 0.5, 0.5], side: 'double',
    }));
    this.type = 'Trail';
    this.frustumCulled = false; // bounds change every frame
    this.target = this;
    this.maxPoints = maxPoints;
    this.width = options.width ?? 0.2;
    this.widthTail = options.widthTail ?? 0;
    this.minDistance = options.minDistance ?? 0.05;
    this.points = new Float32Array(maxPoints * 3);
  }

  /** Clear the history (the ribbon disappears until points accumulate again). */
  reset(): void {
    this.count = 0;
    this.rebuild(_view.set(0, 0, 1));
  }

  /** Sample `target`, append a point if it moved far enough, and rebuild the ribbon. */
  update(cameraPosition: Vector3): void {
    this.target.getWorldPosition(_p);
    if (this.count === 0) {
      this.push(_p);
    } else {
      const h = (this.count - 1) * 3;
      const dx = _p.x - this.points[h], dy = _p.y - this.points[h + 1], dz = _p.z - this.points[h + 2];
      if (dx * dx + dy * dy + dz * dz >= this.minDistance * this.minDistance) this.push(_p);
      else { this.points[h] = _p.x; this.points[h + 1] = _p.y; this.points[h + 2] = _p.z; } // slide the head
    }
    this.rebuild(cameraPosition);
  }

  private push(p: Vector3): void {
    if (this.count < this.maxPoints) {
      const i = this.count * 3;
      this.points[i] = p.x; this.points[i + 1] = p.y; this.points[i + 2] = p.z;
      this.count++;
    } else {
      this.points.copyWithin(0, 3); // drop the oldest
      const i = (this.maxPoints - 1) * 3;
      this.points[i] = p.x; this.points[i + 1] = p.y; this.points[i + 2] = p.z;
    }
  }

  private rebuild(cameraPosition: Vector3): void {
    const n = this.count;
    const geom = this.geometry;
    const pos = geom.attributes.position.array;
    const nrm = geom.attributes.normal.array;
    const uv = geom.attributes.uv.array;
    const pts = this.points;
    const eps = 1e-6;

    for (let i = 0; i < n; i++) {
      _p.fromArray(pts, i * 3);
      // Tangent along the trail (central difference at interior points).
      if (n === 1) _dir.set(0, 0, 0);
      else if (i === 0) _dir.fromArray(pts, 3).sub(_p);
      else if (i === n - 1) _dir.copy(_p).sub(_q.fromArray(pts, (i - 1) * 3));
      else _dir.fromArray(pts, (i + 1) * 3).sub(_q.fromArray(pts, (i - 1) * 3));

      _view.subVectors(cameraPosition, _p);
      _side.crossVectors(_dir, _view);
      if (_side.length() < eps) {
        _side.crossVectors(_dir, _up.set(0, 1, 0));
        if (_side.length() < eps) _side.set(1, 0, 0);
      }
      _side.normalize();
      _view.normalize();

      const t = n > 1 ? i / (n - 1) : 1;                              // 0 oldest → 1 newest
      const w = (this.widthTail + (this.width - this.widthTail) * t) * 0.5;
      const li = i * 6, ri = li + 3;
      pos[li] = _p.x - _side.x * w; pos[li + 1] = _p.y - _side.y * w; pos[li + 2] = _p.z - _side.z * w;
      pos[ri] = _p.x + _side.x * w; pos[ri + 1] = _p.y + _side.y * w; pos[ri + 2] = _p.z + _side.z * w;
      nrm[li] = _view.x; nrm[li + 1] = _view.y; nrm[li + 2] = _view.z;
      nrm[ri] = _view.x; nrm[ri + 1] = _view.y; nrm[ri + 2] = _view.z;
      const ti = i * 4;
      uv[ti] = t; uv[ti + 1] = 0; uv[ti + 2] = t; uv[ti + 3] = 1;
    }

    // Collapse unused vertex pairs onto the newest point so the spare index
    // triangles are zero-area (colinear) and never render.
    const cx = n > 0 ? pts[(n - 1) * 3] : 0, cy = n > 0 ? pts[(n - 1) * 3 + 1] : 0, cz = n > 0 ? pts[(n - 1) * 3 + 2] : 0;
    for (let i = n; i < this.maxPoints; i++) {
      const li = i * 6, ri = li + 3;
      pos[li] = cx; pos[li + 1] = cy; pos[li + 2] = cz;
      pos[ri] = cx; pos[ri + 1] = cy; pos[ri + 2] = cz;
    }

    geom.version++; // force the renderer to re-upload the rewritten buffers
  }
}
