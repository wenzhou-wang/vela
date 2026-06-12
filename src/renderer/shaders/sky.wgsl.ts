import { FRAME_DEFS } from './pbr.wgsl';

/**
 * Skybox pass: a fullscreen triangle at depth 1 (depthCompare less-equal, no
 * depth write) drawn after the opaques, so only background pixels shade. The
 * vertex stage reconstructs the world-space view ray per corner from the
 * projection diagonal and the transposed view rotation (perspective cameras);
 * the fragment samples the raw equirectangular environment, optionally at a
 * blurred mip (frame.clusterParams.w in 0..1 scales the max mip level).
 *
 * Uses its own bind group (frame uniform + raw env texture) rather than the
 * frame group, because frame binding 4 holds the GGX-prefiltered specular map
 * when IBL is active — too low-res for a sharp background.
 */
export const SKY_SHADER = FRAME_DEFS + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var skyMap : texture_2d<f32>;
@group(0) @binding(2) var skySampler : sampler;

struct SkyOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) ray : vec3<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vi : u32) -> SkyOut {
  var out : SkyOut;
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = 1.0 - f32(vi & 2u) * 2.0;
  out.clip = vec4<f32>(x, y, 1.0, 1.0);
  // NDC corner -> view-space ray (perspective: divide by the projection diagonal),
  // then rotate into world space with the transposed view rotation.
  let dView = vec3<f32>(x / frame.proj[0].x, y / frame.proj[1].y, -1.0);
  let viewRot = mat3x3<f32>(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
  out.ray = transpose(viewRot) * dView;
  return out;
}

fn skyDirToEquirectUv(d : vec3<f32>) -> vec2<f32> {
  let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2<f32>(u, v);
}

fn skyAcesFilmic(x : vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn skyLinearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, cutoff);
}

@fragment
fn fs_sky(in : SkyOut) -> @location(0) vec4<f32> {
  let maxMip = f32(textureNumLevels(skyMap) - 1u);
  let lod = clamp(frame.clusterParams.w, 0.0, 1.0) * maxMip;
  let uv = skyDirToEquirectUv(normalize(in.ray));
  var color = textureSampleLevel(skyMap, skySampler, uv, lod).rgb * frame.envParams.y;
  color = color * frame.ambient.w; // exposure
  // envParams.w bit 0: linear output (post pipeline tonemaps later).
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    color = skyAcesFilmic(color);
    color = skyLinearToSRGB(color);
  }
  return vec4<f32>(color, 1.0);
}
`;
