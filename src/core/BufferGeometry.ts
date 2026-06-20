import { BufferAttribute, type TypedArray } from './BufferAttribute';
import { Box3 } from '../math/Box3';
import { Sphere } from '../math/Sphere';
import { Vector3 } from '../math/Vector3';
import { generateUUID } from '../math/MathUtils';

const _meshletPoint = new Vector3();

export type AttributeName = 'position' | 'normal' | 'uv' | 'tangent' | 'color' | string;
export interface Meshlet { firstIndex: number; indexCount: number; boundingSphere: Sphere }

/** Holds vertex attributes and an optional index buffer for a mesh. */
export class BufferGeometry {
  readonly id: string = generateUUID();
  readonly isBufferGeometry = true;
  name = '';

  attributes: Record<string, BufferAttribute> = {};
  index: BufferAttribute | null = null;

  /**
   * Morph-target deltas. Each entry in `position`/`normal` is one target's
   * per-vertex offset from the base attribute (same length/itemSize). Blend
   * weights live on the `Mesh` (`morphTargetInfluences`).
   */
  morphAttributes: { position?: BufferAttribute[]; normal?: BufferAttribute[] } = {};

  boundingBox: Box3 | null = null;
  boundingSphere: Sphere | null = null;
  /** Optional sequential index ranges for GPU per-meshlet culling. */
  meshlets: Meshlet[] | null = null;
  meshletMaxTriangles = 0;

  /** Incremented when buffers are structurally replaced (forces GPU re-create). */
  version = 0;

  setAttribute(name: AttributeName, attribute: BufferAttribute): this {
    this.attributes[name] = attribute;
    if (name === 'position') this.meshlets = null;
    return this;
  }

  getAttribute(name: AttributeName): BufferAttribute | undefined {
    return this.attributes[name];
  }

  setIndex(index: BufferAttribute | TypedArray): this {
    if (index instanceof BufferAttribute) {
      this.index = index;
    } else {
      this.index = new BufferAttribute(index, 1);
    }
    this.meshlets = null;
    return this;
  }

  /** Number of vertices to draw (index count if indexed, else position count). */
  getDrawCount(): number {
    if (this.index) return this.index.count;
    const pos = this.attributes.position;
    return pos ? pos.count : 0;
  }

  computeBoundingBox(): void {
    const position = this.attributes.position;
    if (!this.boundingBox) this.boundingBox = new Box3();
    if (position) this.boundingBox.setFromBufferArray(position.array);
    else this.boundingBox.makeEmpty();
  }

  computeBoundingSphere(): void {
    const position = this.attributes.position;
    if (!position) return;
    if (!this.boundingBox) this.computeBoundingBox();
    const box = this.boundingBox!;
    const center = new Vector3();
    box.getCenter(center);
    let maxSq = 0;
    const arr = position.array;
    const v = new Vector3();
    for (let i = 0; i < arr.length; i += 3) {
      v.set(arr[i], arr[i + 1], arr[i + 2]);
      maxSq = Math.max(maxSq, center.distanceToSquared(v));
    }
    this.boundingSphere = new Sphere(center, Math.sqrt(maxSq));
  }

  /** Partition indexed triangles into bounded sequential ranges (no index reordering). */
  computeMeshlets(maxTriangles = 64): Meshlet[] {
    if (!this.index) throw new Error('BufferGeometry.computeMeshlets: indexed geometry is required.');
    if (!Number.isInteger(maxTriangles) || maxTriangles < 1) throw new Error('BufferGeometry.computeMeshlets: maxTriangles must be a positive integer.');
    const positions = this.attributes.position?.array;
    if (!positions) throw new Error('BufferGeometry.computeMeshlets: position attribute is required.');
    const indices = this.index.array;
    if (indices.length % 3 !== 0) throw new Error('BufferGeometry.computeMeshlets: index count must be divisible by 3.');
    const result: Meshlet[] = [];
    const chunk = maxTriangles * 3;
    for (let first = 0; first < indices.length; first += chunk) {
      const count = Math.min(chunk, indices.length - first);
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = first; i < first + count; i++) {
        const o = indices[i] * 3, x = positions[o], y = positions[o + 1], z = positions[o + 2];
        minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
        maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
      }
      const center = new Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
      let radiusSq = 0;
      for (let i = first; i < first + count; i++) {
        const o = indices[i] * 3;
        _meshletPoint.set(positions[o], positions[o + 1], positions[o + 2]);
        radiusSq = Math.max(radiusSq, center.distanceToSquared(_meshletPoint));
      }
      result.push({ firstIndex: first, indexCount: count, boundingSphere: new Sphere(center, Math.sqrt(radiusSq)) });
    }
    this.meshlets = result;
    this.meshletMaxTriangles = maxTriangles;
    return result;
  }

  /** Generate smooth vertex normals from positions (indexed or non-indexed). */
  computeVertexNormals(): void {
    const positionAttr = this.attributes.position;
    if (!positionAttr) return;
    const positions = positionAttr.array;
    const normals = new Float32Array(positions.length);

    const pA = new Vector3(), pB = new Vector3(), pC = new Vector3();
    const cb = new Vector3(), ab = new Vector3();

    const addNormal = (ia: number, ib: number, ic: number) => {
      pA.fromArray(positions, ia * 3);
      pB.fromArray(positions, ib * 3);
      pC.fromArray(positions, ic * 3);
      cb.subVectors(pC, pB);
      ab.subVectors(pA, pB);
      cb.cross(ab);
      normals[ia * 3] += cb.x; normals[ia * 3 + 1] += cb.y; normals[ia * 3 + 2] += cb.z;
      normals[ib * 3] += cb.x; normals[ib * 3 + 1] += cb.y; normals[ib * 3 + 2] += cb.z;
      normals[ic * 3] += cb.x; normals[ic * 3 + 1] += cb.y; normals[ic * 3 + 2] += cb.z;
    };

    if (this.index) {
      const idx = this.index.array;
      for (let i = 0; i < idx.length; i += 3) addNormal(idx[i], idx[i + 1], idx[i + 2]);
    } else {
      for (let i = 0; i < positions.length / 3; i += 3) addNormal(i, i + 1, i + 2);
    }

    const v = new Vector3();
    for (let i = 0; i < normals.length; i += 3) {
      v.set(normals[i], normals[i + 1], normals[i + 2]).normalize();
      normals[i] = v.x; normals[i + 1] = v.y; normals[i + 2] = v.z;
    }
    this.setAttribute('normal', new BufferAttribute(normals, 3));
  }
}
