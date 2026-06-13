/**
 * GPU frustum + (optional) hi-Z occlusion cull compute shader.
 * One thread per draw slot: tests the world-space bounding sphere against the
 * six camera frustum planes, then — when occlusion is enabled — against the
 * previous frame's hi-Z max-depth pyramid. Writes instanceCount = 0 (culled)
 * or 1 (visible) into the indirect draw buffer.
 *
 * Bind group 0:
 *   0 – CullParams uniform   (planes, draw count, prev viewProj, hi-Z params)
 *   1 – spheres storage      (one vec4 per slot: center.xyz, radius; radius < 0 = unused)
 *   2 – draws storage r/w     (GPUDrawIndexedIndirectParameters per slot)
 *   3 – hi-Z pyramid texture  (r32float max-depth; mip 0 = full res)
 *   4 – hi-Z sampler          (non-filtering; we use textureLoad)
 */
export const CULL_SHADER = /* wgsl */ `
struct CullParams {
  planes      : array<vec4<f32>, 6>,
  prevViewProj: mat4x4<f32>,   // previous frame's view-projection (matches the hi-Z)
  drawCount   : u32,
  occlusion   : u32,           // 0 = frustum only, 1 = also hi-Z occlusion
  hizMips     : u32,           // mip count of the hi-Z pyramid
  _pad        : u32,
  hizSize     : vec2<f32>,     // mip-0 dimensions (pixels)
  _pad2       : vec2<f32>,
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
@group(0) @binding(3) var hiz : texture_2d<f32>;
@group(0) @binding(4) var hizSmp : sampler;

// Conservative hi-Z occlusion test using the previous frame's max-depth
// pyramid. Returns true if the sphere is definitely behind the stored
// occluders within its screen footprint. Errs toward "visible" (never culls
// when unsure: behind camera, clipping the near plane, or out of the prior view).
fn occluded(s : Sphere) -> bool {
  let c = cull.prevViewProj * vec4<f32>(s.center, 1.0);
  if (c.w <= 0.0) { return false; }
  // Clip-space sphere extent (approx): radius scaled by 1/w.
  let invW = 1.0 / c.w;
  let ndc = c.xyz * invW;
  let rNdc = s.radius * invW;
  // Nearest point of the sphere in NDC depth (smaller = closer in [0,1] depth).
  let nearDepth = ndc.z - rNdc;
  if (nearDepth <= 0.0) { return false; } // crosses near plane → keep
  // Screen-space AABB in UV (y flipped).
  let minUv = vec2<f32>((ndc.x - rNdc) * 0.5 + 0.5, 0.5 - (ndc.y + rNdc) * 0.5);
  let maxUv = vec2<f32>((ndc.x + rNdc) * 0.5 + 0.5, 0.5 - (ndc.y - rNdc) * 0.5);
  if (maxUv.x < 0.0 || minUv.x > 1.0 || maxUv.y < 0.0 || minUv.y > 1.0) { return false; }
  // Pick a mip where the AABB spans ~1-2 texels, so 4 taps cover it.
  let sizePx = (maxUv - minUv) * cull.hizSize;
  let dim = max(max(sizePx.x, sizePx.y), 1.0);
  let mip = clamp(ceil(log2(dim)), 0.0, f32(cull.hizMips - 1u));
  let cl = clamp(vec2<f32>(0.0), vec2<f32>(1.0), minUv);
  let ch = clamp(vec2<f32>(0.0), vec2<f32>(1.0), maxUv);
  // Max occluder depth across the AABB corners at the chosen mip.
  let d0 = textureSampleLevel(hiz, hizSmp, cl, mip).r;
  let d1 = textureSampleLevel(hiz, hizSmp, vec2<f32>(ch.x, cl.y), mip).r;
  let d2 = textureSampleLevel(hiz, hizSmp, vec2<f32>(cl.x, ch.y), mip).r;
  let d3 = textureSampleLevel(hiz, hizSmp, ch, mip).r;
  let maxOccluder = max(max(d0, d1), max(d2, d3));
  // Occluded if the sphere's nearest point is farther than every occluder.
  return nearDepth > maxOccluder;
}

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
  if (visible && cull.occlusion == 1u && occluded(s)) {
    visible = false;
  }
  draws[idx].instanceCount = select(0u, 1u, visible);
}
`;
