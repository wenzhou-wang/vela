/**
 * SSAO (screen-space ambient occlusion) computation pass.
 *
 * Bind group 0:
 *   binding 0 — texture_depth_2d  (scene depth, sampleCount = 1)
 *   binding 1 — sampler           (filtering, non-comparison)
 *   binding 2 — SSAOUniforms      (invProj, proj, params)
 *
 * Output: r channel = (1 − occlusion).  1 = fully lit, 0 = fully occluded.
 * Uses 8 hemisphere samples rotated per-pixel by a hash-based noise vector.
 */
export const SSAO_SHADER = /* wgsl */ `
struct SSAOUniforms {
  invProj : mat4x4<f32>,
  proj    : mat4x4<f32>,
  params  : vec4<f32>,  // (radius, bias, 1/width, 1/height)
};

@group(0) @binding(0) var depthTex : texture_depth_2d;
@group(0) @binding(1) var samp     : sampler;
@group(0) @binding(2) var<uniform> u : SSAOUniforms;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       uv   : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x    = f32((vi << 1u) & 2u);
  let y    = f32( vi         & 2u);
  out.uv   = vec2<f32>(x, y);
  out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

// Reconstruct view-space position from screen UV + NDC depth.
fn posFromDepth(uv : vec2<f32>, d : f32) -> vec3<f32> {
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
  let vp  = u.invProj * ndc;
  return vp.xyz / vp.w;
}

fn sampleDepth(uv : vec2<f32>) -> f32 {
  let size = vec2<i32>(textureDimensions(depthTex));
  let coord = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
  return textureLoad(depthTex, coord, 0);
}

// 8 hemisphere sample directions (roughly uniform, biased toward the pole).
fn kernelDir(i : i32) -> vec3<f32> {
  switch i {
    case 0 : { return vec3<f32>( 0.538,  0.819,  0.200); }
    case 1 : { return vec3<f32>(-0.666,  0.557,  0.495); }
    case 2 : { return vec3<f32>(-0.170, -0.755,  0.634); }
    case 3 : { return vec3<f32>( 0.712, -0.494,  0.499); }
    case 4 : { return vec3<f32>(-0.005,  0.156,  0.988); }
    case 5 : { return vec3<f32>(-0.677,  0.124,  0.726); }
    case 6 : { return vec3<f32>( 0.369, -0.001,  0.929); }
    default: { return vec3<f32>(-0.106, -0.244,  0.964); }
  }
}

// Hash-based noise vector in [-1,1]^3 for per-pixel kernel rotation.
fn noise3(uv : vec2<f32>) -> vec3<f32> {
  let q = vec2<f32>(dot(uv, vec2<f32>(127.1, 311.7)), dot(uv, vec2<f32>(269.5, 183.3)));
  return fract(sin(q.xyx + q.xyy) * 43758.547) * 2.0 - 1.0;
}

// Rotate a kernel vector from tangent space (normal = +Z) into view space.
fn rotate(k : vec3<f32>, normal : vec3<f32>, noise : vec3<f32>) -> vec3<f32> {
  let t = normalize(noise - normal * dot(noise, normal));
  let b = cross(normal, t);
  return t * k.x + b * k.y + normal * k.z;
}

@fragment
fn fs_ssao(in : VSOut) -> @location(0) vec4<f32> {
  let d = sampleDepth(in.uv);
  if (d >= 0.9999) { return vec4<f32>(1.0); }  // background — no occlusion

  let pos  = posFromDepth(in.uv, d);
  let dx   = u.params.z;
  let dy   = u.params.w;

  // Approximate surface normal from depth gradient (view space).
  let posR = posFromDepth(in.uv + vec2<f32>(dx, 0.0),  sampleDepth(in.uv + vec2<f32>(dx, 0.0)));
  let posU = posFromDepth(in.uv + vec2<f32>(0.0, -dy), sampleDepth(in.uv + vec2<f32>(0.0, -dy)));
  let normal = normalize(cross(posU - pos, posR - pos));

  let noise  = normalize(noise3(in.uv * 1024.0));
  let radius = u.params.x;
  let bias   = u.params.y;

  var occ = 0.0;
  for (var i = 0; i < 8; i++) {
    let dir    = normalize(rotate(kernelDir(i), normal, noise));
    let sp     = pos + dir * radius;

    // Project sample back to screen space.
    let clip4  = u.proj * vec4<f32>(sp, 1.0);
    let suv    = vec2<f32>( clip4.x / clip4.w * 0.5 + 0.5,
                            0.5 - clip4.y / clip4.w * 0.5);
    let sd     = sampleDepth(suv);
    let sv     = posFromDepth(suv, sd);

    // Only count samples within the AO radius (soft range check prevents halos).
    let range  = smoothstep(0.0, 1.0, radius / max(abs(pos.z - sv.z), 1e-4));
    occ += select(0.0, 1.0, sv.z >= sp.z + bias) * range;
  }
  return vec4<f32>(1.0 - occ / 8.0);
}
`;
