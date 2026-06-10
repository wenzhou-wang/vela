/**
 * Clustered forward+ light assignment.
 *
 * The view frustum is divided into a CLUSTER_X × CLUSTER_Y screen-tile grid with
 * CLUSTER_Z logarithmic depth slices. One invocation per cluster computes the
 * cluster's view-space AABB and tests every light's bounding sphere against it,
 * writing up to MAX_PER_CLUSTER light indices (prefixed by a count) into the
 * cluster-light list consumed by the PBR fragment shader.
 *
 * Perspective projections only (the AABB derivation scales near-plane corners
 * along view rays). Constants must match the copies in pbr.wgsl.
 */

export const CLUSTER_SHADER = /* wgsl */ `
const CLUSTER_X = 16u;
const CLUSTER_Y = 9u;
const CLUSTER_Z = 24u;
const CLUSTER_COUNT = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
const MAX_PER_CLUSTER = 32u;

struct ClusterParams {
  view    : mat4x4<f32>,  // world → view
  invProj : mat4x4<f32>,  // inverse projection (unjittered)
  info    : vec4<f32>,    // x = numLights, y = near, z = far, w = unused
};

struct Light {
  positionKind  : vec4<f32>,  // xyz = position, w = kind (0 dir, 1 point, 2 spot)
  directionRange: vec4<f32>,  // xyz = direction, w = range (0 = infinite)
  colorDecay    : vec4<f32>,
  spotParams    : vec4<f32>,
};

@group(0) @binding(0) var<uniform> p : ClusterParams;
@group(0) @binding(1) var<storage, read> lights : array<Light>;
@group(0) @binding(2) var<storage, read_write> clusterLights : array<u32>;

// Unproject an NDC xy point onto the view-space near plane.
fn unprojectNear(ndc : vec2<f32>) -> vec3<f32> {
  let v = p.invProj * vec4(ndc, 0.0, 1.0);
  return v.xyz / v.w;
}

@compute @workgroup_size(64)
fn cs_cluster(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= CLUSTER_COUNT) { return; }
  let cx = idx % CLUSTER_X;
  let cy = (idx / CLUSTER_X) % CLUSTER_Y;
  let cz = idx / (CLUSTER_X * CLUSTER_Y);

  let near = p.info.y;
  let far  = p.info.z;
  // Logarithmic slice bounds (positive distances along -Z).
  let z0 = near * pow(far / near, f32(cz) / f32(CLUSTER_Z));
  let z1 = near * pow(far / near, f32(cz + 1u) / f32(CLUSTER_Z));

  // Tile NDC bounds. cy = 0 is the top row (matches framebuffer pixel rows).
  let x0 = f32(cx)      / f32(CLUSTER_X) * 2.0 - 1.0;
  let x1 = f32(cx + 1u) / f32(CLUSTER_X) * 2.0 - 1.0;
  let y0 = 1.0 - f32(cy + 1u) / f32(CLUSTER_Y) * 2.0;
  let y1 = 1.0 - f32(cy)      / f32(CLUSTER_Y) * 2.0;

  // View-space AABB: diagonal near-plane corners scaled along their view rays
  // to both slice depths (xy unprojection is linear in NDC, so the two
  // diagonal corners carry the componentwise extremes).
  let c0 = unprojectNear(vec2(x0, y0));
  let c1 = unprojectNear(vec2(x1, y1));
  let s0 = z0 / near;
  let s1 = z1 / near;
  let mn = min(min(c0 * s0, c1 * s0), min(c0 * s1, c1 * s1));
  let mx = max(max(c0 * s0, c1 * s0), max(c0 * s1, c1 * s1));

  let base = idx * (MAX_PER_CLUSTER + 1u);
  var count = 0u;
  let numLights = u32(p.info.x);
  for (var i = 0u; i < numLights; i = i + 1u) {
    if (count >= MAX_PER_CLUSTER) { break; }
    let light = lights[i];
    let kind = u32(light.positionKind.w);
    var hit = true;
    // Directional lights and infinite-range lights affect every cluster;
    // ranged point/spot lights use a conservative sphere-vs-AABB test.
    if (kind != 0u) {
      let range = light.directionRange.w;
      if (range > 0.0) {
        let lp = (p.view * vec4(light.positionKind.xyz, 1.0)).xyz;
        let d = lp - clamp(lp, mn, mx);
        hit = dot(d, d) <= range * range;
      }
    }
    if (hit) {
      clusterLights[base + 1u + count] = i;
      count = count + 1u;
    }
  }
  clusterLights[base] = count;
}
`;
