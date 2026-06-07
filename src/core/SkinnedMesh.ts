import { Mesh } from './Mesh';
import type { BufferGeometry } from './BufferGeometry';
import type { Material } from '../materials/Material';
import type { Skeleton } from './Skeleton';

/**
 * A mesh deformed on the GPU by a skeleton. The geometry must carry `joints`
 * (vec4 uint) and `weights` (vec4 float) attributes; the renderer uploads the
 * skeleton's bone matrices and uses the skinned pipeline variant.
 */
export class SkinnedMesh extends Mesh {
  readonly isSkinnedMesh = true;
  skeleton: Skeleton;

  constructor(geometry: BufferGeometry, material: Material | Material[], skeleton: Skeleton) {
    super(geometry, material);
    this.type = 'SkinnedMesh';
    this.skeleton = skeleton;
    // Skinning deforms vertices away from the rest-pose bounds, so don't cull
    // by the static bounding sphere.
    this.frustumCulled = false;
  }
}
