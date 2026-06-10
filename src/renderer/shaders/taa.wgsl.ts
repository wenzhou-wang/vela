/**
 * Temporal anti-aliasing resolve pass.
 *
 * Each frame the main pass is rendered with a sub-pixel Halton jitter baked into
 * the projection matrix.  This shader:
 *   1. Undoes the jitter offset when sampling the current frame.
 *   2. Reconstructs the world-space position from depth and reprojects it through
 *      the previous frame's view-projection (camera-motion reprojection; no
 *      per-object motion vectors).
 *   3. Neighborhood-clamps the history sample to limit ghosting.
 *   4. Blends: mix(clamped_history, current, blend).
 *
 * Requires sampleCount == 1 (the scene depth must be samplable).
 */

export const TAA_SHADER = /* wgsl */ `
struct TAAParams {
  prevViewProj : mat4x4<f32>,  // previous frame's unjittered view-projection
  invViewProj  : mat4x4<f32>,  // inverse of the current frame's unjittered view-projection
  info         : vec4<f32>,    // xy = NDC jitter applied this frame, z = blend factor
};

@group(0) @binding(0) var<uniform> p : TAAParams;
@group(0) @binding(1) var current  : texture_2d<f32>;   // current jittered HDR scene
@group(0) @binding(2) var history  : texture_2d<f32>;   // previous accumulated TAA output
@group(0) @binding(3) var depthTex : texture_depth_2d;  // scene depth (non-MSAA)
@group(0) @binding(4) var smp      : sampler;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       uv   : vec2<f32>,
};

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv   = vec2(x, y);
  out.clip = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

// AABB-clamp the history colour against the 3x3 neighbourhood of the current frame.
fn neighborhoodClamp(uv : vec2<f32>, h : vec3<f32>) -> vec3<f32> {
  let texel = 1.0 / vec2<f32>(textureDimensions(current));
  var lo = h;
  var hi = h;
  for (var dy : i32 = -1; dy <= 1; dy++) {
    for (var dx : i32 = -1; dx <= 1; dx++) {
      let c = textureSampleLevel(current, smp, uv + vec2(f32(dx), f32(dy)) * texel, 0.0).rgb;
      lo = min(lo, c);
      hi = max(hi, c);
    }
  }
  return clamp(h, lo, hi);
}

@fragment fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let uv    = in.uv;
  let jit   = p.info.xy;  // NDC jitter: shift UV by (jit.x/2, -jit.y/2)
  let blend = p.info.z;

  // Sample the current HDR scene at the unjittered position.
  let unjUV = uv - jit * vec2(0.5, -0.5);
  let cur   = textureSampleLevel(current, smp, unjUV, 0.0).rgb;

  // Background (cleared depth): nothing to reproject.
  let depth = textureLoad(depthTex, vec2<i32>(in.clip.xy), 0);
  if (depth >= 1.0) {
    return vec4(cur, 1.0);
  }

  // Reconstruct the world position and reproject into the previous frame.
  let ndcPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
  let worldH = p.invViewProj * ndcPos;
  let world  = worldH.xyz / worldH.w;

  let prevClip = p.prevViewProj * vec4(world, 1.0);
  let prevNDC  = prevClip.xy / prevClip.w;
  let prevUV   = vec2((prevNDC.x + 1.0) * 0.5, (1.0 - prevNDC.y) * 0.5);

  // Off-screen last frame: no usable history.
  if (any(prevUV < vec2(0.0)) || any(prevUV > vec2(1.0)) || prevClip.w <= 0.0) {
    return vec4(cur, 1.0);
  }

  let hist = neighborhoodClamp(uv, textureSampleLevel(history, smp, prevUV, 0.0).rgb);
  return vec4(mix(hist, cur, blend), 1.0);
}
`;
