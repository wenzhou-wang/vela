import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Vector3 } from '../math/Vector3';
import type { PointLight } from '../lights/PointLight';

const _pos = new Vector3();

/** A small octahedron wireframe at the point light's position. */
export class PointLightHelper extends LineSegments {
  readonly light: PointLight;
  private readonly trackColor: boolean;

  constructor(light: PointLight, size = 0.5, color?: number) {
    // Octahedron: 6 vertices on the axes, 12 edges.
    const r = size;
    const px = [r, 0, 0], nx = [-r, 0, 0];
    const py = [0, r, 0], ny = [0, -r, 0];
    const pz = [0, 0, r], nz = [0, 0, -r];
    const edges = [
      [py, px], [py, pz], [py, nx], [py, nz], // top fan
      [ny, px], [ny, pz], [ny, nx], [ny, nz], // bottom fan
      [px, pz], [pz, nx], [nx, nz], [nz, px], // equator ring
    ];
    const positions = new Float32Array(edges.length * 2 * 3);
    let o = 0;
    for (const [a, b] of edges) {
      positions[o++] = a[0]; positions[o++] = a[1]; positions[o++] = a[2];
      positions[o++] = b[0]; positions[o++] = b[1]; positions[o++] = b[2];
    }
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(positions, 3));

    const mat = new LineBasicMaterial(color !== undefined ? { color } : {});
    super(geometry, mat);
    this.type = 'PointLightHelper';
    this.frustumCulled = false;
    this.light = light;
    this.trackColor = color === undefined;
    this.update();
  }

  /** Re-position the gizmo at the light and refresh its color. */
  update(): void {
    this.light.updateWorldMatrix(true, false);
    _pos.setFromMatrixPosition(this.light.matrixWorld);
    this.position.copy(_pos);
    if (this.trackColor) (this.material as LineBasicMaterial).color.copy(this.light.color);
  }
}
