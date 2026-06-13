/**
 * Hi-Z depth pyramid construction.
 *
 * Pass "copy" samples the scene depth (texture_depth_2d) into mip 0 of an
 * r32float pyramid. Pass "down" max-reduces a 2×2 block of the previous mip
 * into the next — a conservative max so a sphere is culled only if it is behind
 * the farthest occluder in its screen footprint. Each mip is a separate
 * fullscreen draw into that mip's render target.
 */

export const HIZ_COPY_SHADER = /* wgsl */ `
@group(0) @binding(0) var depthTex : texture_depth_2d;

struct VSOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv = vec2(x, y);
  out.clip = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

@fragment fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let dim = textureDimensions(depthTex);
  let px = clamp(vec2<i32>(in.uv * vec2<f32>(dim)), vec2<i32>(0), vec2<i32>(dim) - vec2<i32>(1));
  return vec4<f32>(textureLoad(depthTex, px, 0), 0.0, 0.0, 1.0);
}
`;

export const HIZ_DOWN_SHADER = /* wgsl */ `
@group(0) @binding(0) var srcTex : texture_2d<f32>;

struct VSOut { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv = vec2(x, y);
  out.clip = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

// Max of the 2x2 source texels covering this destination texel (conservative
// for "is the nearest point of a bounding sphere behind all of these?").
@fragment fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let srcDim = textureDimensions(srcTex);
  let base = vec2<i32>(in.uv * vec2<f32>(srcDim));
  let m = vec2<i32>(srcDim) - vec2<i32>(1);
  let d0 = textureLoad(srcTex, clamp(base + vec2<i32>(0, 0), vec2<i32>(0), m), 0).r;
  let d1 = textureLoad(srcTex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), m), 0).r;
  let d2 = textureLoad(srcTex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), m), 0).r;
  let d3 = textureLoad(srcTex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), m), 0).r;
  return vec4<f32>(max(max(d0, d1), max(d2, d3)), 0.0, 0.0, 1.0);
}
`;
