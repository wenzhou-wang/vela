import { Mesh } from './Mesh';
import type { BufferGeometry } from './BufferGeometry';
import type { Material } from '../materials/Material';
import type { Matrix4 } from '../math/Matrix4';

/**
 * Renders one geometry many times in a single draw call. Each instance has its
 * own model matrix stored in `instanceMatrix` (16 floats per instance); the
 * renderer uploads them to a storage buffer read by the instanced vertex shader.
 */
export class InstancedMesh extends Mesh {
  readonly isInstancedMesh = true;
  count: number;
  instanceMatrix: Float32Array<ArrayBuffer>;
  /** Bump (or call `needsUpdate`) after editing matrices to re-upload. */
  version = 0;

  constructor(geometry: BufferGeometry, material: Material | Material[], count: number) {
    super(geometry, material);
    this.type = 'InstancedMesh';
    this.count = count;
    this.instanceMatrix = new Float32Array(count * 16);
    // Instances spread through space; the single geometry bound doesn't describe them.
    this.frustumCulled = false;

    // initialize to identity
    for (let i = 0; i < count; i++) {
      this.instanceMatrix[i * 16] = 1;
      this.instanceMatrix[i * 16 + 5] = 1;
      this.instanceMatrix[i * 16 + 10] = 1;
      this.instanceMatrix[i * 16 + 15] = 1;
    }
  }

  setMatrixAt(index: number, matrix: Matrix4): void {
    this.instanceMatrix.set(matrix.elements, index * 16);
  }

  getMatrixAt(index: number, target: Matrix4): void {
    target.fromArray(this.instanceMatrix, index * 16);
  }

  needsUpdate(): void {
    this.version++;
  }
}
