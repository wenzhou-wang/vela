import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';

/** A box centered at the origin. */
export class BoxGeometry extends BufferGeometry {
  constructor(width = 1, height = 1, depth = 1) {
    super();
    const hw = width / 2, hh = height / 2, hd = depth / 2;

    const positions: number[] = [];
    const normals: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    let offset = 0;

    const face = (
      nx: number, ny: number, nz: number,
      corners: [number, number, number][],
    ) => {
      for (let i = 0; i < 4; i++) {
        positions.push(...corners[i]);
        normals.push(nx, ny, nz);
      }
      uvs.push(0, 1, 1, 1, 1, 0, 0, 0);
      indices.push(offset, offset + 1, offset + 2, offset, offset + 2, offset + 3);
      offset += 4;
    };

    // +X
    face(1, 0, 0, [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]]);
    // -X
    face(-1, 0, 0, [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]]);
    // +Y
    face(0, 1, 0, [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]]);
    // -Y
    face(0, -1, 0, [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]]);
    // +Z
    face(0, 0, 1, [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]]);
    // -Z
    face(0, 0, -1, [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]]);

    this.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
    this.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
    this.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
    this.setIndex(new Uint16Array(indices));
  }
}
