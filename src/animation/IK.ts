import { Vector3 } from '../math/Vector3';
import { Quaternion } from '../math/Quaternion';
import { Matrix4 } from '../math/Matrix4';
import type { Object3D } from '../core/Object3D';

const _a = new Vector3(), _b = new Vector3(), _c = new Vector3(), _t = new Vector3();
const _d = new Vector3(), _perp = new Vector3(), _bend = new Vector3();
const _bDes = new Vector3(), _tEff = new Vector3(), _from = new Vector3(), _to = new Vector3(), _ax = new Vector3();
const _m = new Matrix4();
const _wq = new Quaternion(), _pq = new Quaternion(), _R = new Quaternion();

const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));
const worldPos = (o: Object3D, out: Vector3): void => { out.setFromMatrixPosition(o.matrixWorld); };
const worldQuat = (o: Object3D, out: Quaternion): void => { _m.extractRotation(o.matrixWorld); out.setFromRotationMatrix(_m); };

/** Apply a world-space rotation `R` to `obj` (about its origin) as a local quaternion. */
function premulWorld(obj: Object3D, R: Quaternion): void {
  worldQuat(obj, _wq);
  if (obj.parent) { _m.extractRotation(obj.parent.matrixWorld); _pq.setFromRotationMatrix(_m); }
  else _pq.set(0, 0, 0, 1);
  // local = inv(parentWorld) · R · worldOld
  obj.quaternion.copy(_pq).invert().multiply(R).multiply(_wq).normalize();
}

export interface TwoBoneIKOptions {
  /** World-space hint the joint bends toward (knee/elbow). Omit to keep the current bend plane. */
  pole?: Vector3;
}

/**
 * Analytic two-bone IK: rotate `root` and `mid` (local quaternions) so `end`
 * reaches `target` (world space), with the joint bending toward an optional
 * `pole`. `mid` must be a child of `root` and `end` a descendant of `mid`.
 *
 * Geometric solve — place the mid joint on the law-of-cosines circle in the
 * plane spanned by the aim axis and the pole, then orient each bone to its
 * desired direction. Robust to a fully straight starting pose. Apply after the
 * `AnimationMixer` writes the pose (it overrides the two joint rotations); an
 * out-of-reach target extends the limb straight. Pure math — offline-verifiable.
 */
export function solveTwoBoneIK(root: Object3D, mid: Object3D, end: Object3D, target: Vector3, options: TwoBoneIKOptions = {}): void {
  root.updateWorldMatrix(true, true);
  worldPos(root, _a); worldPos(mid, _b); worldPos(end, _c); _t.copy(target);

  const eps = 1e-5;
  const lab = _b.distanceTo(_a);
  const lcb = _b.distanceTo(_c);
  if (lab < eps || lcb < eps) return; // degenerate bones
  const dist = _a.distanceTo(_t);
  const lat = clamp(dist, Math.abs(lab - lcb) + eps, lab + lcb - eps);

  // Aim axis a→t (fall back to the current limb direction if target sits on the root).
  if (dist < eps) _d.subVectors(_c, _a).normalize();
  else _d.subVectors(_t, _a).multiplyScalar(1 / dist);

  // Bend direction: component of the pole (or the current mid) perpendicular to the aim axis.
  if (options.pole) _perp.subVectors(options.pole, _a);
  else _perp.subVectors(_b, _a);
  _perp.sub(_bend.copy(_d).multiplyScalar(_perp.dot(_d)));
  if (_perp.length() < eps) {
    // Pole/limb colinear with aim: pick any axis perpendicular to the aim direction.
    _ax.set(Math.abs(_d.x) < 0.9 ? 1 : 0, Math.abs(_d.x) < 0.9 ? 0 : 1, 0);
    _perp.crossVectors(_d, _ax);
  }
  _bend.copy(_perp).normalize();

  // Desired mid position on the law-of-cosines circle; desired end at the clamped reach.
  const cosT = clamp((lab * lab + lat * lat - lcb * lcb) / (2 * lab * lat), -1, 1);
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  _bDes.copy(_a)
    .add(_from.copy(_d).multiplyScalar(lab * cosT))
    .add(_to.copy(_bend).multiplyScalar(lab * sinT));
  _tEff.copy(_a).add(_from.copy(_d).multiplyScalar(lat));

  // Rotate root so the upper bone (a→b) points at the desired mid position.
  _from.subVectors(_b, _a).normalize();
  _to.subVectors(_bDes, _a).normalize();
  premulWorld(root, _R.setFromUnitVectors(_from, _to));
  root.updateWorldMatrix(true, true);

  // Rotate mid so the lower bone (b→c) points at the (clamped) target.
  worldPos(end, _c);
  _from.subVectors(_c, _bDes).normalize();
  _to.subVectors(_tEff, _bDes).normalize();
  premulWorld(mid, _R.setFromUnitVectors(_from, _to));
  root.updateWorldMatrix(true, true);
}
