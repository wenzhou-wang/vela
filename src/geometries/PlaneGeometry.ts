import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';

/** A flat plane on the XZ axes (normal +Y), centered at the origin. */
export class PlaneGeometry extends BufferGeometry {
  constructor(width = 1, depth = 1) {
    super();
    const hw = width / 2, hd = depth / 2;
    const positions = new Float32Array([
      -hw, 0, hd, hw, 0, hd, hw, 0, -hd, -hw, 0, -hd,
    ]);
    const normals = new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]);
    const uvs = new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    this.setAttribute('position', new BufferAttribute(positions, 3));
    this.setAttribute('normal', new BufferAttribute(normals, 3));
    this.setAttribute('uv', new BufferAttribute(uvs, 2));
    this.setIndex(indices);
  }
}
