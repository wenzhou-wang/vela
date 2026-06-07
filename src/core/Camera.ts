import { Object3D } from './Object3D';
import { Matrix4 } from '../math/Matrix4';

/** Base camera: holds projection and the inverse-world (view) matrix. */
export class Camera extends Object3D {
  readonly isCamera = true;
  readonly projectionMatrix = new Matrix4();
  readonly projectionMatrixInverse = new Matrix4();
  /** View matrix = inverse(world matrix). */
  readonly matrixWorldInverse = new Matrix4();

  constructor() {
    super();
    this.type = 'Camera';
  }

  override updateMatrixWorld(force?: boolean): void {
    super.updateMatrixWorld(force);
    this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }

  override updateWorldMatrix(updateParents: boolean, updateChildren: boolean): void {
    super.updateWorldMatrix(updateParents, updateChildren);
    this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
}
