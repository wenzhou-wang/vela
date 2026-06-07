import { Vector3 } from './Vector3';
import type { Quaternion } from './Quaternion';

const _v1 = new Vector3();
const _zAxis = new Vector3();
const _xAxis = new Vector3();
const _yAxis = new Vector3();

/**
 * A 4x4 matrix stored in column-major order (compatible with WGSL/GLSL).
 * elements[0..3] = column 0, etc.
 */
export class Matrix4 {
  elements: Float32Array;

  constructor() {
    // identity
    this.elements = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  }

  identity(): this {
    const e = this.elements;
    e[0] = 1; e[4] = 0; e[8] = 0; e[12] = 0;
    e[1] = 0; e[5] = 1; e[9] = 0; e[13] = 0;
    e[2] = 0; e[6] = 0; e[10] = 1; e[14] = 0;
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    return this;
  }

  copy(m: Matrix4): this {
    this.elements.set(m.elements);
    return this;
  }

  clone(): Matrix4 {
    return new Matrix4().copy(this);
  }

  fromArray(a: ArrayLike<number>, offset = 0): this {
    for (let i = 0; i < 16; i++) this.elements[i] = a[i + offset];
    return this;
  }

  multiply(m: Matrix4): this {
    return this.multiplyMatrices(this, m);
  }

  premultiply(m: Matrix4): this {
    return this.multiplyMatrices(m, this);
  }

  multiplyMatrices(a: Matrix4, b: Matrix4): this {
    const ae = a.elements, be = b.elements, te = this.elements;
    const a11 = ae[0], a12 = ae[4], a13 = ae[8], a14 = ae[12];
    const a21 = ae[1], a22 = ae[5], a23 = ae[9], a24 = ae[13];
    const a31 = ae[2], a32 = ae[6], a33 = ae[10], a34 = ae[14];
    const a41 = ae[3], a42 = ae[7], a43 = ae[11], a44 = ae[15];

    const b11 = be[0], b12 = be[4], b13 = be[8], b14 = be[12];
    const b21 = be[1], b22 = be[5], b23 = be[9], b24 = be[13];
    const b31 = be[2], b32 = be[6], b33 = be[10], b34 = be[14];
    const b41 = be[3], b42 = be[7], b43 = be[11], b44 = be[15];

    te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
    te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
    te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
    te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

    te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
    te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
    te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
    te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

    te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
    te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
    te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
    te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

    te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
    te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
    te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
    te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
    return this;
  }

  compose(position: Vector3, q: Quaternion, scale: Vector3): this {
    const te = this.elements;
    const x = q.x, y = q.y, z = q.z, w = q.w;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;
    const sx = scale.x, sy = scale.y, sz = scale.z;

    te[0] = (1 - (yy + zz)) * sx;
    te[1] = (xy + wz) * sx;
    te[2] = (xz - wy) * sx;
    te[3] = 0;

    te[4] = (xy - wz) * sy;
    te[5] = (1 - (xx + zz)) * sy;
    te[6] = (yz + wx) * sy;
    te[7] = 0;

    te[8] = (xz + wy) * sz;
    te[9] = (yz - wx) * sz;
    te[10] = (1 - (xx + yy)) * sz;
    te[11] = 0;

    te[12] = position.x;
    te[13] = position.y;
    te[14] = position.z;
    te[15] = 1;
    return this;
  }

  decompose(position: Vector3, quaternion: Quaternion, scale: Vector3): this {
    const te = this.elements;
    let sx = _v1.set(te[0], te[1], te[2]).length();
    const sy = _v1.set(te[4], te[5], te[6]).length();
    const sz = _v1.set(te[8], te[9], te[10]).length();

    const det = this.determinant();
    if (det < 0) sx = -sx;

    position.x = te[12];
    position.y = te[13];
    position.z = te[14];

    _m.copy(this);
    const me = _m.elements;
    const invSX = 1 / sx, invSY = 1 / sy, invSZ = 1 / sz;
    me[0] *= invSX; me[1] *= invSX; me[2] *= invSX;
    me[4] *= invSY; me[5] *= invSY; me[6] *= invSY;
    me[8] *= invSZ; me[9] *= invSZ; me[10] *= invSZ;

    quaternion.setFromRotationMatrix(_m);

    scale.x = sx;
    scale.y = sy;
    scale.z = sz;
    return this;
  }

  determinant(): number {
    const e = this.elements;
    const n11 = e[0], n12 = e[4], n13 = e[8], n14 = e[12];
    const n21 = e[1], n22 = e[5], n23 = e[9], n24 = e[13];
    const n31 = e[2], n32 = e[6], n33 = e[10], n34 = e[14];
    const n41 = e[3], n42 = e[7], n43 = e[11], n44 = e[15];

    return (
      n41 * (n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34) +
      n42 * (n11 * n23 * n34 - n11 * n24 * n33 + n14 * n21 * n33 - n13 * n21 * n34 + n13 * n24 * n31 - n14 * n23 * n31) +
      n43 * (n11 * n24 * n32 - n11 * n22 * n34 - n14 * n21 * n32 + n12 * n21 * n34 + n14 * n22 * n31 - n12 * n24 * n31) +
      n44 * (-n13 * n22 * n31 - n11 * n23 * n32 + n11 * n22 * n33 + n13 * n21 * n32 - n12 * n21 * n33 + n12 * n23 * n31)
    );
  }

  invert(): this {
    const te = this.elements;
    const n11 = te[0], n21 = te[1], n31 = te[2], n41 = te[3];
    const n12 = te[4], n22 = te[5], n32 = te[6], n42 = te[7];
    const n13 = te[8], n23 = te[9], n33 = te[10], n43 = te[11];
    const n14 = te[12], n24 = te[13], n34 = te[14], n44 = te[15];

    const t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    const t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    const t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    const t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    if (det === 0) return this.identity();
    const idet = 1 / det;

    te[0] = t11 * idet;
    te[1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    te[2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    te[3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    te[4] = t12 * idet;
    te[5] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    te[6] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    te[7] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    te[8] = t13 * idet;
    te[9] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    te[10] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    te[11] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    te[12] = t14 * idet;
    te[13] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    te[14] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    te[15] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;
    return this;
  }

  transpose(): this {
    const e = this.elements;
    let t;
    t = e[1]; e[1] = e[4]; e[4] = t;
    t = e[2]; e[2] = e[8]; e[8] = t;
    t = e[6]; e[6] = e[9]; e[9] = t;
    t = e[3]; e[3] = e[12]; e[12] = t;
    t = e[7]; e[7] = e[13]; e[13] = t;
    t = e[11]; e[11] = e[14]; e[14] = t;
    return this;
  }

  /** Builds a right-handed view matrix looking from `eye` toward `target`. */
  lookAt(eye: Vector3, target: Vector3, up: Vector3): this {
    _zAxis.subVectors(eye, target);
    if (_zAxis.lengthSq() === 0) _zAxis.z = 1;
    _zAxis.normalize();

    _xAxis.crossVectors(up, _zAxis);
    if (_xAxis.lengthSq() === 0) {
      if (Math.abs(up.z) === 1) _zAxis.x += 0.0001;
      else _zAxis.z += 0.0001;
      _zAxis.normalize();
      _xAxis.crossVectors(up, _zAxis);
    }
    _xAxis.normalize();
    _yAxis.crossVectors(_zAxis, _xAxis);

    const e = this.elements;
    e[0] = _xAxis.x; e[4] = _yAxis.x; e[8] = _zAxis.x;
    e[1] = _xAxis.y; e[5] = _yAxis.y; e[9] = _zAxis.y;
    e[2] = _xAxis.z; e[6] = _yAxis.z; e[10] = _zAxis.z;
    return this;
  }

  /**
   * Right-handed perspective projection mapping z to [0, 1] (WebGPU NDC).
   */
  makePerspective(fovYRadians: number, aspect: number, near: number, far: number): this {
    const f = 1.0 / Math.tan(fovYRadians / 2);
    const nf = 1 / (near - far);
    const e = this.elements;
    e[0] = f / aspect; e[4] = 0; e[8] = 0; e[12] = 0;
    e[1] = 0; e[5] = f; e[9] = 0; e[13] = 0;
    e[2] = 0; e[6] = 0; e[10] = far * nf; e[14] = far * near * nf;
    e[3] = 0; e[7] = 0; e[11] = -1; e[15] = 0;
    return this;
  }

  /** Right-handed orthographic projection mapping z to [0, 1] (WebGPU NDC). */
  makeOrthographic(left: number, right: number, top: number, bottom: number, near: number, far: number): this {
    const e = this.elements;
    const w = 1.0 / (right - left);
    const h = 1.0 / (top - bottom);
    const p = 1.0 / (far - near);
    e[0] = 2 * w; e[4] = 0; e[8] = 0; e[12] = -(right + left) * w;
    e[1] = 0; e[5] = 2 * h; e[9] = 0; e[13] = -(top + bottom) * h;
    e[2] = 0; e[6] = 0; e[10] = -p; e[14] = -near * p;
    e[3] = 0; e[7] = 0; e[11] = 0; e[15] = 1;
    return this;
  }

  setPosition(v: Vector3): this {
    const e = this.elements;
    e[12] = v.x;
    e[13] = v.y;
    e[14] = v.z;
    return this;
  }

  extractRotation(m: Matrix4): this {
    const te = this.elements, me = m.elements;
    const sx = 1 / _v1.set(me[0], me[1], me[2]).length();
    const sy = 1 / _v1.set(me[4], me[5], me[6]).length();
    const sz = 1 / _v1.set(me[8], me[9], me[10]).length();
    te[0] = me[0] * sx; te[1] = me[1] * sx; te[2] = me[2] * sx; te[3] = 0;
    te[4] = me[4] * sy; te[5] = me[5] * sy; te[6] = me[6] * sy; te[7] = 0;
    te[8] = me[8] * sz; te[9] = me[9] * sz; te[10] = me[10] * sz; te[11] = 0;
    te[12] = 0; te[13] = 0; te[14] = 0; te[15] = 1;
    return this;
  }
}

const _m = new Matrix4();
