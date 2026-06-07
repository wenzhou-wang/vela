import { Vector3 } from '../math/Vector3';
import { Quaternion } from '../math/Quaternion';
import { Euler } from '../math/Euler';
import { Matrix4 } from '../math/Matrix4';
import { generateUUID } from '../math/MathUtils';

const _q = new Quaternion();
const _m1 = new Matrix4();
const _target = new Vector3();
const _position = new Vector3();

/** Base node of the scene graph: a transform with a parent and children. */
export class Object3D {
  readonly id: string = generateUUID();
  readonly isObject3D = true;
  name = '';
  type = 'Object3D';

  parent: Object3D | null = null;
  readonly children: Object3D[] = [];

  readonly position = new Vector3(0, 0, 0);
  readonly rotation = new Euler(0, 0, 0);
  readonly quaternion = new Quaternion();
  readonly scale = new Vector3(1, 1, 1);

  readonly matrix = new Matrix4();
  readonly matrixWorld = new Matrix4();

  /** When true, local matrix is recomposed from position/quaternion/scale each update. */
  matrixAutoUpdate = true;
  matrixWorldNeedsUpdate = false;
  visible = true;

  constructor() {
    // keep rotation (Euler) and quaternion in sync
    this.rotation.onChange = () => {
      this.quaternion.setFromEuler(this.rotation);
    };
  }

  add(...objects: Object3D[]): this {
    for (const object of objects) {
      if (object === this) continue;
      if (object.parent) object.parent.remove(object);
      object.parent = this;
      this.children.push(object);
    }
    return this;
  }

  remove(...objects: Object3D[]): this {
    for (const object of objects) {
      const index = this.children.indexOf(object);
      if (index !== -1) {
        object.parent = null;
        this.children.splice(index, 1);
      }
    }
    return this;
  }

  removeFromParent(): this {
    if (this.parent) this.parent.remove(this);
    return this;
  }

  /** Depth-first traversal including self. */
  traverse(callback: (object: Object3D) => void): void {
    callback(this);
    const children = this.children;
    for (let i = 0; i < children.length; i++) {
      children[i].traverse(callback);
    }
  }

  traverseVisible(callback: (object: Object3D) => void): void {
    if (!this.visible) return;
    callback(this);
    const children = this.children;
    for (let i = 0; i < children.length; i++) {
      children[i].traverseVisible(callback);
    }
  }

  getWorldPosition(target: Vector3): Vector3 {
    this.updateWorldMatrix(true, false);
    return target.setFromMatrixPosition(this.matrixWorld);
  }

  /** Orient this object so its -Z axis points toward the target (world space). */
  lookAt(x: number | Vector3, y?: number, z?: number): void {
    if (x instanceof Vector3) _target.copy(x);
    else _target.set(x, y!, z!);

    this.updateWorldMatrix(true, false);
    _position.setFromMatrixPosition(this.matrixWorld);

    // cameras/lights look down -Z; everything else down +Z
    const isCameraOrLight = this.type.includes('Camera') || this.type.includes('Light');
    if (isCameraOrLight) _m1.lookAt(_position, _target, this.up);
    else _m1.lookAt(_target, _position, this.up);

    this.quaternion.setFromRotationMatrix(_m1);

    if (this.parent) {
      _m1.extractRotation(this.parent.matrixWorld);
      _q.setFromRotationMatrix(_m1);
      this.quaternion.premultiply(_q.invert());
    }
  }

  up = new Vector3(0, 1, 0);

  updateMatrix(): void {
    this.matrix.compose(this.position, this.quaternion, this.scale);
    this.matrixWorldNeedsUpdate = true;
  }

  /** Recompute world matrices for this subtree. */
  updateMatrixWorld(force = false): void {
    if (this.matrixAutoUpdate) this.updateMatrix();

    if (this.matrixWorldNeedsUpdate || force) {
      if (this.parent === null) {
        this.matrixWorld.copy(this.matrix);
      } else {
        this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
      }
      this.matrixWorldNeedsUpdate = false;
      force = true;
    }

    const children = this.children;
    for (let i = 0; i < children.length; i++) {
      children[i].updateMatrixWorld(force);
    }
  }

  /** Update world matrix of this node, optionally walking up parents and/or down children. */
  updateWorldMatrix(updateParents: boolean, updateChildren: boolean): void {
    if (updateParents && this.parent) {
      this.parent.updateWorldMatrix(true, false);
    }
    if (this.matrixAutoUpdate) this.updateMatrix();

    if (this.parent === null) {
      this.matrixWorld.copy(this.matrix);
    } else {
      this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
    }

    if (updateChildren) {
      const children = this.children;
      for (let i = 0; i < children.length; i++) {
        children[i].updateWorldMatrix(false, true);
      }
    }
  }
}
