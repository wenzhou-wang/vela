/**
 * Fullscreen post-processing passes. A single oversized triangle generates the
 * UVs; each fragment entry samples the previous pass's color target(s).
 *
 * Module bindings (one shared layout; `bloom` is a dummy for passes that ignore it):
 *   0 src texture · 1 sampler · 2 params uniform · 3 bloom texture · 4 SSAO texture
 *   params.data = (1/width, 1/height, bloomThreshold, bloomIntensity)
 *
 * Entries: tonemap (ACES+sRGB), tonemapBloom (adds blurred bloom then tonemaps),
 * tonemapAgx / tonemapAgxBloom (AgX operator), threshold (bright-pass),
 * blurH/blurV (separable Gaussian), fxaa, copy, oitComposite.
 */
export const POST_SHADER = /* wgsl */ `
@group(0) @binding(0) var src      : texture_2d<f32>;
@group(0) @binding(1) var samp     : sampler;
struct Params {
  data : vec4<f32>,  // (1/width, 1/height, bloomThreshold, bloomIntensity)
  ssao : vec4<f32>,  // (ssaoStrength, unused, unused, unused)
  lift : vec4<f32>,
  gamma : vec4<f32>,
  gain : vec4<f32>,
  grade : vec4<f32>, // x=saturation, y=enabled
  lens : vec4<f32>, // vignette, chromatic aberration, bloom streak, flare
};
@group(0) @binding(2) var<uniform> params   : Params;
@group(0) @binding(3) var bloomTex : texture_2d<f32>;
@group(0) @binding(4) var ssaoTex  : texture_2d<f32>;
struct Exposure { value : vec4<f32> };
@group(0) @binding(5) var<uniform> exposure : Exposure;

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

// AgX tonemap (minimal approximation, after Troy Sobotka's AgX via bwrensch).
// Returns a LINEAR value; the caller applies the sRGB OETF like the ACES path.
fn agxContrast(x : vec3<f32>) -> vec3<f32> {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
    - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agxTonemap(hdr : vec3<f32>) -> vec3<f32> {
  // Inset (input) transform, column-major.
  let inset = mat3x3<f32>(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let outset = mat3x3<f32>(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393;
  let maxEv = 4.026069;

  var v = inset * max(hdr, vec3<f32>(0.0));
  v = clamp(log2(v), vec3<f32>(minEv), vec3<f32>(maxEv));
  v = (v - minEv) / (maxEv - minEv);
  v = agxContrast(v);              // display-encoded sigmoid output
  v = outset * v;                  // outset (output) transform
  return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));  // EOTF → linear
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
fn grade(c : vec3<f32>) -> vec3<f32> {
  if (params.grade.y < 0.5) { return c; }
  var v = pow(max((c + params.lift.rgb) * params.gain.rgb, vec3<f32>(0.0)), vec3<f32>(1.0) / max(params.gamma.rgb, vec3<f32>(1e-4)));
  return mix(vec3<f32>(luma(v)), v, params.grade.x);
}
fn hdrSample(uv:vec2<f32>)->vec3<f32>{
  let d=uv-vec2<f32>(0.5); let off=d*params.lens.y;
  return vec3<f32>(textureSample(src,samp,uv+off).r,textureSample(src,samp,uv).g,textureSample(src,samp,uv-off).b)*exposure.value.x;
}
fn bloomSample(uv:vec2<f32>)->vec3<f32>{
  var b=textureSample(bloomTex,samp,uv).rgb;
  if(params.lens.z>0.0){for(var i=1;i<=3;i=i+1){let o=vec2<f32>(params.data.x*f32(i)*12.0,0.0);b+=0.25*params.lens.z*(textureSample(bloomTex,samp,uv+o).rgb+textureSample(bloomTex,samp,uv-o).rgb);}}
  if(params.lens.w>0.0){b+=textureSample(bloomTex,samp,vec2<f32>(1.0)-uv).rgb*params.lens.w;}
  return b;
}
fn finish(c:vec3<f32>,uv:vec2<f32>)->vec4<f32>{let d=distance(uv,vec2<f32>(0.5))*1.4142;let v=1.0-params.lens.x*smoothstep(0.35,1.0,d);return vec4<f32>(grade(c)*v,1.0);}

@fragment
fn fs_tonemap(in : VSOut) -> @location(0) vec4<f32> {
  let ao  = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr = hdrSample(in.uv) * ao;
  return finish(linearToSRGB(acesFilmic(hdr)),in.uv);
}

@fragment
fn fs_tonemapBloom(in : VSOut) -> @location(0) vec4<f32> {
  let ao    = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr   = hdrSample(in.uv) * ao;
  let bloom = bloomSample(in.uv) * params.data.w;
  return finish(linearToSRGB(acesFilmic(hdr + bloom)),in.uv);
}

@fragment
fn fs_tonemapAgx(in : VSOut) -> @location(0) vec4<f32> {
  let ao  = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr = hdrSample(in.uv) * ao;
  return finish(linearToSRGB(agxTonemap(hdr)),in.uv);
}

@fragment
fn fs_tonemapAgxBloom(in : VSOut) -> @location(0) vec4<f32> {
  let ao    = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr   = hdrSample(in.uv) * ao;
  let bloom = bloomSample(in.uv) * params.data.w;
  return finish(linearToSRGB(agxTonemap(hdr + bloom)),in.uv);
}

// No-tonemap output: exposure already applied upstream; just sRGB-encode (with
// the SSAO multiply, like the tonemap entries). For flat/stylized looks that
// should not get the filmic curve.
@fragment
fn fs_linear(in : VSOut) -> @location(0) vec4<f32> {
  let ao  = mix(1.0, textureSample(ssaoTex, samp, in.uv).r, params.ssao.x);
  let hdr = hdrSample(in.uv) * ao;
  return finish(linearToSRGB(hdr),in.uv);
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
  let mid = textureSampleLevel(src, samp, in.uv, 0.0);
  let lM = luma(mid.rgb);
  let lNW = luma(textureSampleLevel(src, samp, in.uv + vec2<f32>(-rcp.x, -rcp.y), 0.0).rgb);
  let lNE = luma(textureSampleLevel(src, samp, in.uv + vec2<f32>( rcp.x, -rcp.y), 0.0).rgb);
  let lSW = luma(textureSampleLevel(src, samp, in.uv + vec2<f32>(-rcp.x,  rcp.y), 0.0).rgb);
  let lSE = luma(textureSampleLevel(src, samp, in.uv + vec2<f32>( rcp.x,  rcp.y), 0.0).rgb);

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
    textureSampleLevel(src, samp, in.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
    textureSampleLevel(src, samp, in.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
  let rgbB = rgbA * 0.5 + 0.25 * (
    textureSampleLevel(src, samp, in.uv + dir * -0.5, 0.0).rgb +
    textureSampleLevel(src, samp, in.uv + dir * 0.5, 0.0).rgb);
  let lB = luma(rgbB);
  if (lB < lMin || lB > lMax) { return vec4<f32>(rgbA, mid.a); }
  return vec4<f32>(rgbB, mid.a);
}
`;
