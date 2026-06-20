import { Object3D } from './Object3D';
import { Vector3 } from '../math/Vector3';
import { Color } from '../math/Color';
import type { Texture } from '../textures/Texture';

export interface DecalOptions {
  size?: Vector3;
  color?: Color;
  opacity?: number;
  map?: Texture | null;
}

/** A depth-projected box decal. Its local -Z axis is the projection direction. */
export class Decal extends Object3D {
  readonly isDecal = true;
  readonly size: Vector3;
  readonly color: Color;
  opacity: number;
  map: Texture | null;

  constructor(options: DecalOptions = {}) {
    super();
    this.type = 'Decal';
    this.size = options.size?.clone() ?? new Vector3(1, 1, 1);
    this.color = options.color?.clone() ?? new Color(1, 1, 1);
    this.opacity = options.opacity ?? 1;
    this.map = options.map ?? null;
  }
}
