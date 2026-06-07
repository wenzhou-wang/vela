/**
 * Fullscreen post-processing passes. A single oversized triangle generates the
 * UVs; each fragment entry samples the previous pass's color target.
 *
 * - `fs_tonemap` — ACES filmic + linear→sRGB (moved out of the material shader so
 *   the scene can render to a linear HDR target first).
 * - `fs_fxaa` — cheap edge antialiasing on the tonemapped (sRGB) image.
 * - `fs_copy` — straight blit.
 *
 * `params`: x = exposure (applied in tonemap), y = 1/width, z = 1/height.
 */
export const POST_SHADER = /* wgsl */ `
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
struct Params { data : vec4<f32> };
@group(0) @binding(2) var<uniform> params : Params;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  // Fullscreen triangle: clip (-1,-1),(3,-1),(-1,3); uv covers [0,1].
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv = vec2<f32>(x, y);
  out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

fn acesFilmic(x : vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, cutoff);
}

@fragment
fn fs_tonemap(in : VSOut) -> @location(0) vec4<f32> {
  let hdr = textureSample(src, samp, in.uv).rgb;
  return vec4<f32>(linearToSRGB(acesFilmic(hdr)), 1.0);
}

@fragment
fn fs_copy(in : VSOut) -> @location(0) vec4<f32> {
  return textureSample(src, samp, in.uv);
}

fn luma(c : vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// Compact FXAA (Lottes-style) operating on the sRGB image.
@fragment
fn fs_fxaa(in : VSOut) -> @location(0) vec4<f32> {
  let rcp = params.data.yz; // (1/width, 1/height)
  let mid = textureSample(src, samp, in.uv);
  let lM = luma(mid.rgb);
  let lNW = luma(textureSample(src, samp, in.uv + vec2<f32>(-rcp.x, -rcp.y)).rgb);
  let lNE = luma(textureSample(src, samp, in.uv + vec2<f32>( rcp.x, -rcp.y)).rgb);
  let lSW = luma(textureSample(src, samp, in.uv + vec2<f32>(-rcp.x,  rcp.y)).rgb);
  let lSE = luma(textureSample(src, samp, in.uv + vec2<f32>( rcp.x,  rcp.y)).rgb);

  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  if (lMax - lMin < 0.0625) { return mid; } // no significant edge

  var dir = vec2<f32>(
    -((lNW + lNE) - (lSW + lSE)),
     ((lNW + lSW) - (lNE + lSE)),
  );
  let reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  let rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
  dir = clamp(dir * rcpDir, vec2<f32>(-8.0), vec2<f32>(8.0)) * rcp;

  let rgbA = 0.5 * (
    textureSample(src, samp, in.uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    textureSample(src, samp, in.uv + dir * (2.0 / 3.0 - 0.5)).rgb);
  let rgbB = rgbA * 0.5 + 0.25 * (
    textureSample(src, samp, in.uv + dir * -0.5).rgb +
    textureSample(src, samp, in.uv + dir * 0.5).rgb);
  let lB = luma(rgbB);
  if (lB < lMin || lB > lMax) { return vec4<f32>(rgbA, mid.a); }
  return vec4<f32>(rgbB, mid.a);
}
`;
