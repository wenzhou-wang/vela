import { Object3D } from './Object3D';
import type { BufferGeometry } from './BufferGeometry';
import type { Material } from '../materials/Material';

/** A renderable: geometry paired with one (or more) materials. */
export class Mesh extends Object3D {
  readonly isMesh = true;
  geometry: BufferGeometry;
  material: Material | Material[];

  constructor(geometry: BufferGeometry, material: Material | Material[]) {
    super();
    this.type = 'Mesh';
    this.geometry = geometry;
    this.material = material;
  }
}
