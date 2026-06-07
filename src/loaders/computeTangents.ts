import type { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';

/**
 * Computes per-vertex tangents (vec4 with handedness in w) from position,
 * normal and uv, using Lengyel's method. Required for tangent-space normal
 * mapping when a glTF mesh has a normal map but no TANGENT attribute.
 */
export function computeTangents(geometry: BufferGeometry): void {
  const positionAttr = geometry.getAttribute('position');
  const normalAttr = geometry.getAttribute('normal');
  const uvAttr = geometry.getAttribute('uv');
  if (!positionAttr || !normalAttr || !uvAttr) return;

  const positions = positionAttr.array;
  const normals = normalAttr.array;
  const uvs = uvAttr.array;
  const vertexCount = positionAttr.count;

  const indices: ArrayLike<number> = geometry.index
    ? geometry.index.array
    : sequence(vertexCount);
  const triCount = indices.length / 3;

  const tan1 = new Float32Array(vertexCount * 3);
  const tan2 = new Float32Array(vertexCount * 3);

  for (let t = 0; t < triCount; t++) {
    const i0 = indices[t * 3];
    const i1 = indices[t * 3 + 1];
    const i2 = indices[t * 3 + 2];

    const x0 = positions[i0 * 3], y0 = positions[i0 * 3 + 1], z0 = positions[i0 * 3 + 2];
    const x1 = positions[i1 * 3], y1 = positions[i1 * 3 + 1], z1 = positions[i1 * 3 + 2];
    const x2 = positions[i2 * 3], y2 = positions[i2 * 3 + 1], z2 = positions[i2 * 3 + 2];

    const u0 = uvs[i0 * 2], v0 = uvs[i0 * 2 + 1];
    const u1 = uvs[i1 * 2], v1 = uvs[i1 * 2 + 1];
    const u2 = uvs[i2 * 2], v2 = uvs[i2 * 2 + 1];

    const e1x = x1 - x0, e1y = y1 - y0, e1z = z1 - z0;
    const e2x = x2 - x0, e2y = y2 - y0, e2z = z2 - z0;
    const du1 = u1 - u0, dv1 = v1 - v0;
    const du2 = u2 - u0, dv2 = v2 - v0;

    const denom = du1 * dv2 - du2 * dv1;
    const r = denom === 0 ? 0 : 1 / denom;

    const sx = (dv2 * e1x - dv1 * e2x) * r;
    const sy = (dv2 * e1y - dv1 * e2y) * r;
    const sz = (dv2 * e1z - dv1 * e2z) * r;
    const tx = (du1 * e2x - du2 * e1x) * r;
    const ty = (du1 * e2y - du2 * e1y) * r;
    const tz = (du1 * e2z - du2 * e1z) * r;

    for (const i of [i0, i1, i2]) {
      tan1[i * 3] += sx; tan1[i * 3 + 1] += sy; tan1[i * 3 + 2] += sz;
      tan2[i * 3] += tx; tan2[i * 3 + 1] += ty; tan2[i * 3 + 2] += tz;
    }
  }

  const tangents = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    const nx = normals[i * 3], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];
    let tx = tan1[i * 3], ty = tan1[i * 3 + 1], tz = tan1[i * 3 + 2];

    // Gram-Schmidt orthogonalize tangent against normal
    const ndt = nx * tx + ny * ty + nz * tz;
    tx -= nx * ndt; ty -= ny * ndt; tz -= nz * ndt;
    let len = Math.hypot(tx, ty, tz);
    if (len < 1e-8) { tx = 1; ty = 0; tz = 0; len = 1; }
    tx /= len; ty /= len; tz /= len;

    // handedness
    const cx = ny * tz - nz * ty;
    const cy = nz * tx - nx * tz;
    const cz = nx * ty - ny * tx;
    const w = cx * tan2[i * 3] + cy * tan2[i * 3 + 1] + cz * tan2[i * 3 + 2] < 0 ? -1 : 1;

    tangents[i * 4] = tx;
    tangents[i * 4 + 1] = ty;
    tangents[i * 4 + 2] = tz;
    tangents[i * 4 + 3] = w;
  }

  geometry.setAttribute('tangent', new BufferAttribute(tangents, 4));
}

function sequence(n: number): Uint32Array {
  const a = new Uint32Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  return a;
}
