/**
 * Fullscreen post-processing passes. A single oversized triangle generates the
 * UVs; each fragment entry samples the previous pass's color target(s).
 *
 * Module bindings (one shared layout; `bloom` is a dummy for passes that ignore it):
 *   0 src texture · 1 sampler · 2 params uniform · 3 bloom texture
 *   params.data = (1/width, 1/height, bloomThreshold, bloomIntensity)
 *
 * Entries: tonemap (ACES+sRGB), tonemapBloom (adds blurred bloom then tonemaps),
 * threshold (bright-pass), blurH/blurV (separable Gaussian), fxaa, copy.
 */
export const POST_SHADER = /* wgsl */ `
@group(0) @binding(0) var src      : texture_2d<f32>;
@group(0) @binding(1) var samp     : sampler;
struct Params {
  data : vec4<f32>,  // (1/width, 1/height, bloomThreshold, bloomIntensity)
  ssao : vec4<f32>,  // (ssaoStrength, 0, 0, 0)
};
@group(0) @binding(2) var<uniform> params   : Params;
@group(0) @binding(3) var bloomTex : texture_2d<f32>;
@group(0) @binding(4) var ssaoTex  : texture_2d<f32>;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
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

fn luma(c : vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_tonemap(in : VSOut) -> @location(0) vec4<f32> {
  let ao  = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr = textureSample(src, samp, in.uv).rgb * ao;
  return vec4<f32>(linearToSRGB(acesFilmic(hdr)), 1.0);
}

@fragment
fn fs_tonemapBloom(in : VSOut) -> @location(0) vec4<f32> {
  let ao    = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr   = textureSample(src, samp, in.uv).rgb * ao;
  let bloom = textureSample(bloomTex, samp, in.uv).rgb * params.data.w;
  return vec4<f32>(linearToSRGB(acesFilmic(hdr + bloom)), 1.0);
}

@fragment
fn fs_copy(in : VSOut) -> @location(0) vec4<f32> {
  return textureSample(src, samp, in.uv);
}

// Weighted-blended OIT resolve: src = accum (rgb*aw, aw), bloomTex.r = revealage.
// Returned with alpha = revealage; the caller blends (1-srcAlpha, srcAlpha) so the
// result is avgColor*(1-reveal) + dst*reveal.
@fragment
fn fs_oitComposite(in : VSOut) -> @location(0) vec4<f32> {
  let accum = textureSample(src, samp, in.uv);
  let reveal = textureSample(bloomTex, samp, in.uv).r;
  let avg = accum.rgb / max(accum.a, 1e-5);
  return vec4<f32>(avg, reveal);
}

// Bright-pass: keep energy above the threshold (soft knee via the excess ratio).
@fragment
fn fs_threshold(in : VSOut) -> @location(0) vec4<f32> {
  let c = textureSample(src, samp, in.uv).rgb;
  let l = luma(c);
  let thr = params.data.z;
  let contrib = max(l - thr, 0.0) / max(l, 1e-4);
  return vec4<f32>(c * contrib, 1.0);
}

// 9-tap separable Gaussian (weights 0.227/0.316/0.070 over offsets 0,1.385,3.231).
const W0 = 0.2270270270;
const W1 = 0.3162162162;
const W2 = 0.0702702703;
const O1 = 1.3846153846;
const O2 = 3.2307692308;

fn blur(uv : vec2<f32>, dir : vec2<f32>) -> vec3<f32> {
  var c = textureSample(src, samp, uv).rgb * W0;
  c = c + textureSample(src, samp, uv + dir * O1).rgb * W1;
  c = c + textureSample(src, samp, uv - dir * O1).rgb * W1;
  c = c + textureSample(src, samp, uv + dir * O2).rgb * W2;
  c = c + textureSample(src, samp, uv - dir * O2).rgb * W2;
  return c;
}

@fragment
fn fs_blurH(in : VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(blur(in.uv, vec2<f32>(params.data.x, 0.0)), 1.0);
}

@fragment
fn fs_blurV(in : VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(blur(in.uv, vec2<f32>(0.0, params.data.y)), 1.0);
}

// Compact FXAA (Lottes-style) operating on the sRGB image.
@fragment
fn fs_fxaa(in : VSOut) -> @location(0) vec4<f32> {
  let rcp = params.data.xy; // (1/width, 1/height)
  let mid = textureSample(src, samp, in.uv);
  let lM = luma(mid.rgb);
  let lNW = luma(textureSample(src, samp, in.uv + vec2<f32>(-rcp.x, -rcp.y)).rgb);
  let lNE = luma(textureSample(src, samp, in.uv + vec2<f32>( rcp.x, -rcp.y)).rgb);
  let lSW = luma(textureSample(src, samp, in.uv + vec2<f32>(-rcp.x,  rcp.y)).rgb);
  let lSE = luma(textureSample(src, samp, in.uv + vec2<f32>( rcp.x,  rcp.y)).rgb);

  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  if (lMax - lMin < 0.0625) { return mid; }

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
