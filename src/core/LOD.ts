import { Object3D } from './Object3D';
import { Vector3 } from '../math/Vector3';

interface Level {
  object: Object3D;
  /** Camera distance at/above which this level is used. */
  distance: number;
  /** Switching deadband (world units) to prevent boundary popping. */
  hysteresis: number;
}

/**
 * Level-of-detail node: shows one of several child representations based on
 * distance to the camera, so distant objects render cheaper geometry. Add
 * levels nearest-first; the renderer calls `update()` each frame (when
 * `autoUpdate`) to pick the active level by toggling child visibility, so LOD
 * composes with frustum culling and every other traversal for free.
 *
 * ```ts
 * const lod = new LOD();
 * lod.addLevel(highPolyMesh, 0);     // < 15 m
 * lod.addLevel(midPolyMesh, 15);     // 15–50 m
 * lod.addLevel(lowPolyMesh, 50);     // > 50 m
 * lod.position.set(x, y, z);
 * scene.add(lod);
 * ```
 */
export class LOD extends Object3D {
  readonly isLOD = true;
  readonly levels: Level[] = [];
  /** Let the renderer pick the level each frame. Set false to control manually. */
  autoUpdate = true;
  /** Index of the currently shown level (-1 = none). */
  private current = -1;

  constructor() {
    super();
    this.type = 'LOD';
  }

  /**
   * Add a level shown at/beyond `distance` (world units). `object` is parented
   * to this LOD. `hysteresis` (world units) widens the switch boundary to
   * avoid flicker when the camera hovers at a threshold.
   */
  addLevel(object: Object3D, distance = 0, hysteresis = 0): this {
    this.add(object);
    object.visible = false;
    const level: Level = { object, distance: Math.abs(distance), hysteresis: Math.abs(hysteresis) };
    // Keep levels sorted by ascending distance.
    let i = 0;
    while (i < this.levels.length && this.levels[i].distance <= level.distance) i++;
    this.levels.splice(i, 0, level);
    return this;
  }

  /** Select the active level for `cameraPosition`, toggling child visibility. */
  update(cameraPosition: Vector3): void {
    if (this.levels.length === 0) return;
    this.getWorldPosition(_pos);
    const d = _pos.distanceTo(cameraPosition);

    let next: number;
    if (this.current < 0) {
      // Cold start: pick the natural level (highest threshold <= d), no deadband.
      next = 0;
      for (let i = 1; i < this.levels.length; i++) {
        if (d >= this.levels[i].distance) next = i;
        else break;
      }
    } else {
      // Hysteresis only resists leaving the current level: switch up only past
      // the next level's distance + hysteresis, down only below this level's
      // distance - hysteresis.
      next = this.current;
      while (next + 1 < this.levels.length &&
             d >= this.levels[next + 1].distance + this.levels[next + 1].hysteresis) {
        next++;
      }
      while (next > 0 && d < this.levels[next].distance - this.levels[next].hysteresis) {
        next--;
      }
    }

    if (next !== this.current && this.current >= 0) {
      this.levels[this.current].object.visible = false;
    }
    this.levels[next].object.visible = true;
    this.current = next;
  }

  /** The currently active level's object, or null. */
  get activeLevel(): Object3D | null {
    return this.current >= 0 ? this.levels[this.current].object : null;
  }
}

const _pos = new Vector3();
