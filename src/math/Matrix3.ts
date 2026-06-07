import type { Matrix4 } from './Matrix4';

/** A 3x3 matrix (column-major), used mainly for normal matrices. */
export class Matrix3 {
  elements: Float32Array;

  constructor() {
    this.elements = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  }

  /** Extract the upper-left 3x3 of a Matrix4. */
  setFromMatrix4(m: Matrix4): this {
    const me = m.elements, te = this.elements;
    te[0] = me[0]; te[1] = me[1]; te[2] = me[2];
    te[3] = me[4]; te[4] = me[5]; te[5] = me[6];
    te[6] = me[8]; te[7] = me[9]; te[8] = me[10];
    return this;
  }

  invert(): this {
    const te = this.elements;
    const n11 = te[0], n21 = te[1], n31 = te[2];
    const n12 = te[3], n22 = te[4], n32 = te[5];
    const n13 = te[6], n23 = te[7], n33 = te[8];
    const t11 = n33 * n22 - n32 * n23;
    const t12 = n32 * n13 - n33 * n12;
    const t13 = n23 * n12 - n22 * n13;
    const det = n11 * t11 + n21 * t12 + n31 * t13;
    if (det === 0) {
      te[0] = 1; te[1] = 0; te[2] = 0;
      te[3] = 0; te[4] = 1; te[5] = 0;
      te[6] = 0; te[7] = 0; te[8] = 1;
      return this;
    }
    const idet = 1 / det;
    te[0] = t11 * idet;
    te[1] = (n31 * n23 - n33 * n21) * idet;
    te[2] = (n32 * n21 - n31 * n22) * idet;
    te[3] = t12 * idet;
    te[4] = (n33 * n11 - n31 * n13) * idet;
    te[5] = (n31 * n12 - n32 * n11) * idet;
    te[6] = t13 * idet;
    te[7] = (n21 * n13 - n23 * n11) * idet;
    te[8] = (n22 * n11 - n21 * n12) * idet;
    return this;
  }

  transpose(): this {
    const m = this.elements;
    let t;
    t = m[1]; m[1] = m[3]; m[3] = t;
    t = m[2]; m[2] = m[6]; m[6] = t;
    t = m[5]; m[5] = m[7]; m[7] = t;
    return this;
  }

  /** Normal matrix = transpose(inverse(upper-left 3x3 of modelMatrix)). */
  getNormalMatrix(m: Matrix4): this {
    return this.setFromMatrix4(m).invert().transpose();
  }
}
