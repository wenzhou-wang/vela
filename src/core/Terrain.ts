import { Object3D } from './Object3D';
import { LOD } from './LOD';
import { Mesh } from './Mesh';
import { BufferGeometry } from './BufferGeometry';
import { BufferAttribute } from './BufferAttribute';
import type { Material } from '../materials/Material';

export interface TerrainOptions {
  /** Row-major `(segmentsX + 1) × (segmentsZ + 1)` height samples. */
  heights: Float32Array;
  segmentsX: number;
  segmentsZ: number;
  width?: number;
  depth?: number;
  tiles?: [number, number];
  levels?: number;
  material: Material;
  hysteresis?: number;
}

export interface TerrainDescription {
  segments: [number, number];
  tiles: [number, number];
  levels: number;
  trianglesByLevel: number[];
}

/** A tiled heightfield whose tiles independently select ordinary `LOD` mesh levels. */
export class Terrain extends Object3D {
  readonly isTerrain = true;
  readonly tiles: LOD[] = [];
  readonly options: Readonly<TerrainOptions>;
  private readonly trianglesByLevel: number[];

  constructor(options: TerrainOptions) {
    super();
    this.type = 'Terrain';
    const { segmentsX: sx, segmentsZ: sz } = options;
    const [tx, tz] = options.tiles ?? [4, 4];
    const levels = options.levels ?? 3;
    if (![sx, sz, tx, tz, levels].every((n) => Number.isInteger(n) && n > 0)) {
      throw new Error('Terrain: segments, tile counts, and levels must be positive integers.');
    }
    if (sx % tx !== 0 || sz % tz !== 0) {
      throw new Error(`Terrain: segments ${sx}x${sz} must be divisible by tiles ${tx}x${tz}.`);
    }
    if (options.heights.length !== (sx + 1) * (sz + 1)) {
      throw new Error(`Terrain: expected ${(sx + 1) * (sz + 1)} height samples, got ${options.heights.length}.`);
    }
    this.options = options;
    this.trianglesByLevel = new Array(levels).fill(0);
    const width = options.width ?? sx;
    const depth = options.depth ?? sz;
    const tileSegX = sx / tx, tileSegZ = sz / tz;
    const tileWidth = width / tx, tileDepth = depth / tz;
    for (let z = 0; z < tz; z++) {
      for (let x = 0; x < tx; x++) {
        const lod = new LOD();
        lod.type = 'TerrainTile';
        lod.position.set(-width / 2 + (x + 0.5) * tileWidth, 0, -depth / 2 + (z + 0.5) * tileDepth);
        for (let level = 0; level < levels; level++) {
          const segX = Math.max(1, tileSegX >> level);
          const segZ = Math.max(1, tileSegZ >> level);
          const geometry = this.buildTile(x, z, segX, segZ, tileSegX, tileSegZ, tileWidth, tileDepth);
          const mesh = new Mesh(geometry, options.material);
          mesh.name = `terrain-${x}-${z}-lod${level}`;
          const distance = level === 0 ? 0 : Math.max(tileWidth, tileDepth) * Math.pow(2, level);
          lod.addLevel(mesh, distance, options.hysteresis ?? Math.max(tileWidth, tileDepth) * 0.1);
          this.trianglesByLevel[level] += segX * segZ * 2;
        }
        this.tiles.push(lod);
        this.add(lod);
      }
    }
  }

  describe(): TerrainDescription {
    return {
      segments: [this.options.segmentsX, this.options.segmentsZ],
      tiles: this.options.tiles ?? [4, 4],
      levels: this.options.levels ?? 3,
      trianglesByLevel: [...this.trianglesByLevel],
    };
  }

  private buildTile(tileX: number, tileZ: number, segX: number, segZ: number,
    baseX: number, baseZ: number, width: number, depth: number): BufferGeometry {
    const positions = new Float32Array((segX + 1) * (segZ + 1) * 3);
    const uvs = new Float32Array((segX + 1) * (segZ + 1) * 2);
    let p = 0, u = 0;
    for (let z = 0; z <= segZ; z++) {
      for (let x = 0; x <= segX; x++) {
        const gx = tileX * baseX + Math.round(x / segX * baseX);
        const gz = tileZ * baseZ + Math.round(z / segZ * baseZ);
        positions[p++] = x / segX * width - width / 2;
        positions[p++] = this.options.heights[gz * (this.options.segmentsX + 1) + gx];
        positions[p++] = z / segZ * depth - depth / 2;
        uvs[u++] = gx / this.options.segmentsX;
        uvs[u++] = gz / this.options.segmentsZ;
      }
    }
    const Index = (segX + 1) * (segZ + 1) > 65535 ? Uint32Array : Uint16Array;
    const indices = new Index(segX * segZ * 6);
    let i = 0;
    for (let z = 0; z < segZ; z++) {
      for (let x = 0; x < segX; x++) {
        const a = z * (segX + 1) + x, b = a + 1, c = a + segX + 1, d = c + 1;
        indices[i++] = a; indices[i++] = c; indices[i++] = b;
        indices[i++] = b; indices[i++] = c; indices[i++] = d;
      }
    }
    const geometry = new BufferGeometry()
      .setAttribute('position', new BufferAttribute(positions, 3))
      .setAttribute('uv', new BufferAttribute(uvs, 2))
      .setIndex(indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }
}
