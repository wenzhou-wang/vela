/**
 * Fullscreen post-processing passes. A single oversized triangle generates the
 * UVs; each fragment entry samples the previous pass's color target(s).
 *
 * Module bindings (one shared layout; `bloom` is a dummy for passes that ignore it):
 *   0 src texture · 1 sampler · 2 params uniform · 3 bloom texture
 *   4 SSAO texture · 5 scene depth
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
  ssao : vec4<f32>,  // (ssaoStrength, unused, outlineThickness, outlineStrength)
  toon : vec4<f32>,  // (depthThreshold, normalThreshold, colorThreshold, outerWidthScale)
  outline : vec4<f32>, // (outline RGB, celEnabled)
  invProj : mat4x4<f32>,
};
@group(0) @binding(2) var<uniform> params   : Params;
@group(0) @binding(3) var bloomTex : texture_2d<f32>;
@group(0) @binding(4) var ssaoTex  : texture_2d<f32>;
@group(0) @binding(5) var depthTex : texture_depth_2d;

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

fn sceneDepth(uv : vec2<f32>) -> f32 {
  let size = vec2<i32>(textureDimensions(depthTex));
  let coord = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
  return textureLoad(depthTex, coord, 0);
}

fn viewPosition(uv : vec2<f32>, depth : f32) -> vec3<f32> {
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
  let view = params.invProj * ndc;
  return view.xyz / view.w;
}

fn isGeometry(depth : f32) -> f32 {
  return select(1.0, 0.0, depth >= 0.9999);
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
fn fs_diffuse(in : VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(linearToSRGB(textureSample(src, samp, in.uv).rgb), 1.0);
}

@fragment
fn fs_copy(in : VSOut) -> @location(0) vec4<f32> {
  return textureSample(src, samp, in.uv);
}

// Bitmoji-style cel treatment. Silhouettes come from depth occupancy at a
// wider radius; interior lines combine relative depth, depth-derived normals,
// and a Sobel color edge so texture-authored facial details stay visible.
@fragment
fn fs_cel(in : VSOut) -> @location(0) vec4<f32> {
  let center = textureSample(src, samp, in.uv);
  let innerStep = params.data.xy * max(params.ssao.z, 0.5);
  let outerStep = innerStep * max(params.toon.w, 1.0);

  let dC = sceneDepth(in.uv);
  let object = isGeometry(dC);

  let oL  = isGeometry(sceneDepth(in.uv + vec2<f32>(-outerStep.x, 0.0)));
  let oR  = isGeometry(sceneDepth(in.uv + vec2<f32>( outerStep.x, 0.0)));
  let oU  = isGeometry(sceneDepth(in.uv + vec2<f32>(0.0, -outerStep.y)));
  let oD  = isGeometry(sceneDepth(in.uv + vec2<f32>(0.0,  outerStep.y)));
  let oUL = isGeometry(sceneDepth(in.uv + vec2<f32>(-outerStep.x, -outerStep.y)));
  let oUR = isGeometry(sceneDepth(in.uv + vec2<f32>( outerStep.x, -outerStep.y)));
  let oDL = isGeometry(sceneDepth(in.uv + vec2<f32>(-outerStep.x,  outerStep.y)));
  let oDR = isGeometry(sceneDepth(in.uv + vec2<f32>( outerStep.x,  outerStep.y)));
  let outerEdge = max(
    max(max(abs(object - oL), abs(object - oR)), max(abs(object - oU), abs(object - oD))),
    max(max(abs(object - oUL), abs(object - oUR)), max(abs(object - oDL), abs(object - oDR))),
  );

  let uvL  = in.uv + vec2<f32>(-innerStep.x, 0.0);
  let uvR  = in.uv + vec2<f32>( innerStep.x, 0.0);
  let uvU  = in.uv + vec2<f32>(0.0, -innerStep.y);
  let uvD  = in.uv + vec2<f32>(0.0,  innerStep.y);
  let uvUL = in.uv + vec2<f32>(-innerStep.x, -innerStep.y);
  let uvUR = in.uv + vec2<f32>( innerStep.x, -innerStep.y);
  let uvDL = in.uv + vec2<f32>(-innerStep.x,  innerStep.y);
  let uvDR = in.uv + vec2<f32>( innerStep.x,  innerStep.y);

  let dL = sceneDepth(uvL); let dR = sceneDepth(uvR);
  let dU = sceneDepth(uvU); let dD = sceneDepth(uvD);
  let dUL = sceneDepth(uvUL); let dUR = sceneDepth(uvUR);
  let dDL = sceneDepth(uvDL); let dDR = sceneDepth(uvDR);
  let pC = viewPosition(in.uv, dC);
  let pL = viewPosition(uvL, dL); let pR = viewPosition(uvR, dR);
  let pU = viewPosition(uvU, dU); let pD = viewPosition(uvD, dD);
  let pUL = viewPosition(uvUL, dUL); let pUR = viewPosition(uvUR, dUR);
  let pDL = viewPosition(uvDL, dDL); let pDR = viewPosition(uvDR, dDR);

  let neighborsAreGeometry = min(min(isGeometry(dL), isGeometry(dR)), min(isGeometry(dU), isGeometry(dD)));
  let depthScale = max(abs(pC.z), 1e-3);
  let relativeDepth = max(
    max(abs(pC.z - pL.z), abs(pC.z - pR.z)),
    max(abs(pC.z - pU.z), abs(pC.z - pD.z)),
  ) / depthScale;
  let depthEdge = smoothstep(params.toon.x, params.toon.x * 3.0, relativeDepth) * object * neighborsAreGeometry;

  let nC = normalize(cross(pR - pL, pD - pU));
  let nL = normalize(cross(pC - pL, pDL - pUL));
  let nR = normalize(cross(pR - pC, pDR - pUR));
  let nU = normalize(cross(pUR - pUL, pC - pU));
  let nD = normalize(cross(pDR - pDL, pD - pC));
  let normalDelta = max(
    max(1.0 - abs(dot(nC, nL)), 1.0 - abs(dot(nC, nR))),
    max(1.0 - abs(dot(nC, nU)), 1.0 - abs(dot(nC, nD))),
  );
  let normalEdge = smoothstep(params.toon.y * 0.5, params.toon.y, normalDelta) * object * neighborsAreGeometry;

  let cL = textureSample(src, samp, uvL).rgb; let cR = textureSample(src, samp, uvR).rgb;
  let cU = textureSample(src, samp, uvU).rgb; let cD = textureSample(src, samp, uvD).rgb;
  let cUL = textureSample(src, samp, uvUL).rgb; let cUR = textureSample(src, samp, uvUR).rgb;
  let cDL = textureSample(src, samp, uvDL).rgb; let cDR = textureSample(src, samp, uvDR).rgb;
  let sobelX = -cUL - 2.0 * cL - cDL + cUR + 2.0 * cR + cDR;
  let sobelY = -cUL - 2.0 * cU - cUR + cDL + 2.0 * cD + cDR;
  let colorDelta = (length(sobelX) + length(sobelY)) * 0.25;
  let colorEdge = smoothstep(params.toon.z, params.toon.z * 2.5, colorDelta) * object;

  let innerEdge = max(depthEdge, max(normalEdge, colorEdge)) * 0.78;
  let edge = clamp(max(outerEdge, innerEdge) * params.ssao.w, 0.0, 1.0);
  return vec4<f32>(mix(center.rgb, params.outline.rgb, edge), center.a);
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
