import { Object3D } from './Object3D';
import { Color } from '../math/Color';
import { Vector2 } from '../math/Vector2';
import type { Texture } from '../textures/Texture';

/**
 * A camera-facing textured quad. World-space by default (sized in world
 * units, depth-tested against the scene — labels, pickups, billboards);
 * set `screenSpace = true` for HUD elements sized in CSS pixels and drawn
 * on top of everything.
 *
 * Sprites sharing the same texture and mode are batched into one instanced
 * draw, so thousands of sprites stay cheap.
 */
export class Sprite extends Object3D {
  readonly isSprite = true;
  /** Color texture (un-set = solid `color` quad). */
  texture: Texture | null = null;
  /** Tint multiplied with the texture. */
  color = new Color(1, 1, 1);
  opacity = 1;
  /** Quad size: world units, or CSS pixels when `screenSpace`. */
  size = new Vector2(1, 1);
  /**
   * Extra offset from the anchor point, in the same units as `size`
   * (x right, y up). Useful to hang a label above its world position.
   */
  offset = new Vector2(0, 0);
  /** Draw as a HUD overlay: no depth test, sized in CSS pixels. */
  screenSpace = false;

  constructor(texture: Texture | null = null) {
    super();
    this.type = 'Sprite';
    this.texture = texture;
  }
}
