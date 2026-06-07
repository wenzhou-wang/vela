import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Vector3 } from '../math/Vector3';
import type { DirectionalLight } from '../lights/DirectionalLight';

const _pos = new Vector3();
const _target = new Vector3();

/**
 * A square at the light's position plus a line pointing toward its target.
 * Add it to the scene root and call {@link update} each frame to track the light.
 */
export class DirectionalLightHelper extends LineSegments {
  readonly light: DirectionalLight;
  /** When false, the gizmo color is fixed; otherwise it follows the light color. */
  private readonly trackColor: boolean;

  constructor(light: DirectionalLight, size = 1, color?: number) {
    const h = size / 2;
    // square in the XY plane (4 edges) + a -Z direction line. `lookAt` (non-camera
    // convention) leaves -Z pointing toward the target, so the line aims at it.
    // prettier-ignore
    const positions = new Float32Array([
      -h, -h, 0,  h, -h, 0,
       h, -h, 0,  h,  h, 0,
       h,  h, 0, -h,  h, 0,
      -h,  h, 0, -h, -h, 0,
       0,  0, 0,  0,  0, -size,
    ]);
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));

    const mat = new LineBasicMaterial(color !== undefined ? { color } : {});
    super(geometry, mat);
    this.type = 'DirectionalLightHelper';
    this.frustumCulled = false;
    this.light = light;
    this.trackColor = color === undefined;
    this.update();
  }

  /** Re-position/orient the gizmo from the light's current transform. */
  update(): void {
    const light = this.light;
    light.updateWorldMatrix(true, false);
    light.target.updateWorldMatrix(true, false);

    _pos.setFromMatrixPosition(light.matrixWorld);
    this.position.copy(_pos);
    _target.setFromMatrixPosition(light.target.matrixWorld);
    this.lookAt(_target); // -Z toward the target (non-camera/light lookAt convention)

    if (this.trackColor) (this.material as LineBasicMaterial).color.copy(light.color);
  }
}
