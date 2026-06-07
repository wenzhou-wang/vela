import type { Object3D } from './Object3D';
import { Matrix4 } from '../math/Matrix4';

const _result = new Matrix4();

/**
 * A set of joint nodes plus their inverse bind matrices. Each frame `update()`
 * recomputes `boneMatrices[i] = jointWorld[i] * inverseBind[i]` (already in
 * world space), which the skinned vertex shader blends by vertex weights.
 */
export class Skeleton {
  joints: Object3D[];
  boneInverses: Matrix4[];
  /** Flat 16-floats-per-joint buffer uploaded to the GPU. */
  boneMatrices: Float32Array<ArrayBuffer>;

  constructor(joints: Object3D[], boneInverses: Matrix4[]) {
    this.joints = joints;
    this.boneInverses = boneInverses;
    this.boneMatrices = new Float32Array(joints.length * 16);
  }

  get jointCount(): number {
    return this.joints.length;
  }

  /** Recompute bone matrices from the joints' current world transforms. */
  update(): void {
    const joints = this.joints;
    const inverses = this.boneInverses;
    const out = this.boneMatrices;
    for (let i = 0; i < joints.length; i++) {
      _result.multiplyMatrices(joints[i].matrixWorld, inverses[i]);
      out.set(_result.elements, i * 16);
    }
  }
}
