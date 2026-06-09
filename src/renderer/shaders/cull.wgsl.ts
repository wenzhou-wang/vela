/**
 * GPU frustum-cull compute shader.
 * One thread per draw slot: tests the world-space bounding sphere against the
 * six camera frustum planes and writes instanceCount = 0 (culled) or 1
 * (visible) into the corresponding entry of the indirect draw buffer.
 *
 * Bind group 0:
 *   0 – CullParams uniform  (frustum planes + draw count)
 *   1 – spheres storage     (one vec4 per slot: center.xyz, radius; radius < 0 = unused)
 *   2 – draws storage r/w   (GPUDrawIndexedIndirectParameters per slot)
 */
export const CULL_SHADER = /* wgsl */ `
struct CullParams {
  planes    : array<vec4<f32>, 6>,
  drawCount : u32,
  _pad      : vec3<u32>,
};

struct Sphere {
  center : vec3<f32>,
  radius : f32,
};

struct DrawCmd {
  indexCount    : u32,
  instanceCount : u32,
  firstIndex    : u32,
  baseVertex    : i32,
  firstInstance : u32,
};

@group(0) @binding(0) var<uniform>            cull    : CullParams;
@group(0) @binding(1) var<storage, read>      spheres : array<Sphere>;
@group(0) @binding(2) var<storage, read_write> draws  : array<DrawCmd>;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= cull.drawCount) { return; }
  let s = spheres[idx];
  if (s.radius < 0.0) {
    draws[idx].instanceCount = 0u;
    return;
  }
  var visible = true;
  for (var i = 0u; i < 6u; i++) {
    let p = cull.planes[i];
    if (dot(s.center, p.xyz) + p.w < -s.radius) {
      visible = false;
      break;
    }
  }
  draws[idx].instanceCount = select(0u, 1u, visible);
}
`;
