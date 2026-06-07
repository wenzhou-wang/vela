import { Object3D } from './Object3D';
import type { BufferGeometry } from './BufferGeometry';
import type { Material } from '../materials/Material';

/** A renderable: geometry paired with one (or more) materials. */
export class Mesh extends Object3D {
  readonly isMesh = true;
  geometry: BufferGeometry;
  material: Material | Material[];

  /**
   * Per-morph-target blend weights, parallel to
   * `geometry.morphAttributes.position`. Empty when the mesh has no targets.
   */
  morphTargetInfluences: number[] = [];
  /** Optional name → target-index map (from glTF `mesh.extras.targetNames`). */
  morphTargetDictionary: Record<string, number> | null = null;

  constructor(geometry: BufferGeometry, material: Material | Material[]) {
    super();
    this.type = 'Mesh';
    this.geometry = geometry;
    this.material = material;
    if (geometry.morphAttributes.position) {
      this.morphTargetInfluences = new Array(geometry.morphAttributes.position.length).fill(0);
    }
  }
}
