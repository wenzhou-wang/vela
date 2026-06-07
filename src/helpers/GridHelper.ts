import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Color } from '../math/Color';

/**
 * A wireframe grid on the XZ plane, centered at the origin. The two center lines
 * (the X and Z axes) use `centerColor`; the rest use `gridColor`.
 */
export class GridHelper extends LineSegments {
  constructor(size = 10, divisions = 10, centerColor = 0x888888, gridColor = 0x444444) {
    const step = size / divisions;
    const half = size / 2;
    const center = new Color().setHex(centerColor);
    const grid = new Color().setHex(gridColor);

    const positions: number[] = [];
    const colors: number[] = [];
    for (let i = 0, k = -half; i <= divisions; i++, k += step) {
      // line parallel to X at z=k, then line parallel to Z at x=k
      positions.push(-half, 0, k, half, 0, k);
      positions.push(k, 0, -half, k, 0, half);
      const c = Math.abs(k) < 1e-6 ? center : grid;
      for (let j = 0; j < 4; j++) colors.push(c.r, c.g, c.b, 1);
    }

    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
    geometry.setAttribute('color', new BufferAttribute(new Float32Array(colors), 4));

    super(geometry, new LineBasicMaterial({ vertexColors: true }));
    this.type = 'GridHelper';
  }
}
