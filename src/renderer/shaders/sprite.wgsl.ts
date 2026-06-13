import { FRAME_DEFS } from './pbr.wgsl';

/**
 * Batched sprite / SDF-glyph rendering: 6 vertices per instance, instances
 * pulled from a storage buffer. Two placement modes share the shader:
 *
 * - world (batch.params.x = 0): camera-facing quad at the instance position,
 *   sized in world units, depth-tested (no write).
 * - screen (batch.params.x = 1): the position projects to NDC, then size and
 *   offset apply in device pixels (depth test off; HUD overlay).
 *
 * batch.params.y = 1 switches the fragment to SDF text mode: alpha comes from
 * a screen-space-antialiased threshold of the texture's alpha channel.
 *
 * Output is "literal color" like the line/helper path: sRGB-encoded unless
 * the post pipeline will tonemap (linear-out bit) — sprites are UI-adjacent
 * and should not be exposure/tonemap-shifted.
 */
export const SPRITE_SHADER = FRAME_DEFS + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame : Frame;

struct SpriteInstance {
  posPad     : vec4<f32>,  // xyz = world position
  sizeOffset : vec4<f32>,  // xy = quad size, zw = offset from anchor (same units)
  color      : vec4<f32>,  // rgb tint + opacity
  uvRect     : vec4<f32>,  // x,y = top-left uv, z,w = bottom-right uv
};

struct BatchParams {
  params : vec4<f32>,  // x = screen-space, y = SDF text mode
};

@group(1) @binding(0) var<storage, read> instances : array<SpriteInstance>;
@group(1) @binding(1) var<uniform> batch : BatchParams;
@group(1) @binding(2) var spriteTex : texture_2d<f32>;
@group(1) @binding(3) var spriteSmp : sampler;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var out : VSOut;
  let inst = instances[ii];
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );
  let c = corners[vi];
  let local = c * inst.sizeOffset.xy * 0.5 + inst.sizeOffset.zw;

  if (batch.params.x > 0.5) {
    // Screen space: project the anchor, then offset in device pixels (+y up).
    let clip0 = frame.proj * frame.view * vec4<f32>(inst.posPad.xyz, 1.0);
    let ndc = clip0.xy / max(clip0.w, 1e-4);
    // Framebuffer size from the cluster tile dims (tile = fb / grid).
    let fb = frame.clusterDims.xy * vec2<f32>(16.0, 9.0);
    out.clip = vec4<f32>(ndc + local / fb * 2.0, 0.0, 1.0);
  } else {
    // World space: camera-facing quad from the view-matrix basis.
    let right = vec3<f32>(frame.view[0].x, frame.view[1].x, frame.view[2].x);
    let up    = vec3<f32>(frame.view[0].y, frame.view[1].y, frame.view[2].y);
    let world = inst.posPad.xyz + right * local.x + up * local.y;
    out.clip = frame.proj * frame.view * vec4<f32>(world, 1.0);
  }

  let t = c * 0.5 + 0.5; // 0..1, +y up
  out.uv = vec2<f32>(
    mix(inst.uvRect.x, inst.uvRect.z, t.x),
    mix(inst.uvRect.w, inst.uvRect.y, t.y), // top of the quad = uvRect.y
  );
  out.color = inst.color;
  return out;
}

fn sLinearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  return mix(c * 12.92, 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055, cutoff);
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let sample = textureSample(spriteTex, spriteSmp, in.uv);
  var alpha : f32;
  var rgb : vec3<f32>;
  if (batch.params.y > 0.5) {
    // SDF text: distance in the alpha channel, antialiased at the 0.5 isoline.
    let d = sample.a;
    let w = max(fwidth(d), 1e-4);
    alpha = smoothstep(0.5 - w, 0.5 + w, d) * in.color.a;
    rgb = in.color.rgb;
  } else {
    alpha = sample.a * in.color.a;
    rgb = in.color.rgb * sample.rgb;
  }
  if (alpha <= 0.001) { discard; }
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    rgb = sLinearToSRGB(rgb);
  }
  return vec4<f32>(rgb * alpha, alpha); // premultiplied
}
`;
