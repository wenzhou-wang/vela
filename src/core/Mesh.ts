import { Object3D } from './Object3D';
import type { BufferGeometry } from './BufferGeometry';
import type { Material } from '../materials/Material';

/**
 * Inverted-hull shell pass for a mesh: after the mesh draws, the same geometry
 * is drawn again with back faces (front-culled), extruded `thickness` world
 * units along the vertex normal, using `material`. This is the generic draw
 * primitive behind outline shells, fur, and selection highlights — the look
 * (flat color, fresnel, fur texture, …) lives in `material` (any
 * `StandardMaterial`/`ShaderMaterial`); the engine only provides the extruded
 * back-face draw. Not supported on `InstancedMesh`.
 */
export interface Shell {
  material: Material;
  /** Extrusion distance along the world normal, in world units. */
  thickness: number;
}

/** A renderable: geometry paired with one (or more) materials. */
export class Mesh extends Object3D {
  readonly isMesh = true;
  geometry: BufferGeometry;
  material: Material | Material[];

  /** Optional inverted-hull overlay draw (outlines, fur, highlights). */
  shell: Shell | null = null;

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
