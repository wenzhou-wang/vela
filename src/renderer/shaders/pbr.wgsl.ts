/**
 * Physically based metallic-roughness shader (glTF 2.0 model).
 * - Cook-Torrance specular: GGX NDF, Smith height-correlated visibility, Schlick Fresnel
 * - Lambertian diffuse
 * - Tangent-space normal mapping (safe with flat default normal map)
 * - ACES-approx filmic tonemap + linear->sRGB on output
 *
 * Single "uber" shader: missing material maps are bound as white/flat defaults,
 * so no per-texture pipeline variants are needed. Four vertex variants share one
 * fragment stage: static (PBR_SHADER), GPU-skinned (PBR_SKINNED_SHADER), instanced
 * (PBR_INSTANCED_SHADER), and morph-target (PBR_MORPH_SHADER).
 */

// Frame-level constants and struct definitions, shared by every pass that
// reads the frame uniform (PBR, ShaderMaterial, sky, ...).
export const FRAME_DEFS = /* wgsl */ `
const PI = 3.141592653589793;
const LIGHT_DIRECTIONAL = 0u;
const LIGHT_POINT = 1u;
const LIGHT_SPOT = 2u;

// Clustered forward+ grid (must match clusters.wgsl).
const CLUSTER_X = 16u;
const CLUSTER_Y = 9u;
const CLUSTER_Z = 24u;
const MAX_PER_CLUSTER = 32u;

struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  cameraPos : vec4<f32>,   // xyz = position, w = numLights
  ambient : vec4<f32>,     // rgb = ambient irradiance, w = exposure
  lightViewProj : mat4x4<f32>, // directional shadow caster's view-projection
  shadowParams : vec4<f32>,    // x = enabled, y = map size, z = normal bias, w = caster light index
  envParams : vec4<f32>,       // x = enabled, y = intensity, z = max mip level, w = flags (bit0=linearOut,bit1=IBL)
  clusterParams : vec4<f32>,   // x = clustered enabled, y = near, z = far, w = skybox background blur
  clusterDims : vec4<f32>,     // xy = cluster tile size in pixels, z = elapsed seconds, w = unused
  fogColor : vec4<f32>,        // rgb = fog color (linear), w = mode (0 none, 1 linear, 2 exp2)
  fogParams : vec4<f32>,       // x = near (linear) or density (exp2), y = far, z = height falloff
  prevViewProj : mat4x4<f32>,
};

struct Light {
  positionKind  : vec4<f32>,  // xyz = position, w = kind
  directionRange: vec4<f32>,  // xyz = direction, w = range
  colorDecay    : vec4<f32>,  // rgb = color, w = decay
  spotParams    : vec4<f32>,  // x = cosInner, y = cosOuter, z = shadow tile (spot: tile, point: first cube-face tile, -1 = none), w = unused
};

struct ShadowTile {
  viewProj : mat4x4<f32>,
  region   : vec4<f32>,  // xy = UV offset in atlas, z = UV scale, w = texel step (1/atlasSize)
};
`;

// Shared: frame defs + group-0 bindings + varyings (the mesh-shading header).
export const HEADER = FRAME_DEFS + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var<storage, read> lights : array<Light>;
@group(0) @binding(2) var shadowMap : texture_depth_2d;
@group(0) @binding(3) var shadowSampler : sampler_comparison;
@group(0) @binding(4) var envMap     : texture_2d<f32>;  // raw env (or IBL prefiltered specular)
@group(0) @binding(5) var envSampler : sampler;
@group(0) @binding(6) var irrMap     : texture_2d<f32>;  // IBL irradiance
@group(0) @binding(7) var irrSampler : sampler;
@group(0) @binding(8) var brdfLUT      : texture_2d<f32>;  // IBL split-sum BRDF LUT
@group(0) @binding(9) var brdfSampler  : sampler;
@group(0) @binding(10) var<storage, read> shadowTiles : array<ShadowTile>;
@group(0) @binding(11) var spotAtlas    : texture_depth_2d;
@group(0) @binding(12) var spotAtlasCmp : sampler_comparison;
@group(0) @binding(13) var sceneCapture : texture_2d<f32>; // opaque HDR snapshot for screen-space refraction
@group(0) @binding(14) var<storage, read> clusterLights : array<u32>; // per-cluster light counts + indices
struct ReflectionProbes {
  positionRadius : array<vec4<f32>, 4>,
  intensity : vec4<f32>,
  info : vec4<f32>, // x = count
};
@group(0) @binding(15) var reflectionMaps : texture_2d_array<f32>;
@group(0) @binding(16) var reflectionSampler : sampler;
@group(0) @binding(17) var<uniform> reflectionProbes : ReflectionProbes;
struct IrradianceGrid {
  origin : vec4<f32>,  // xyz = minimum corner, w = enabled
  spacing : vec4<f32>,
  dims : vec4<f32>,
};
@group(0) @binding(18) var<storage, read> irradianceCoefficients : array<vec4<f32>>;
@group(0) @binding(19) var<uniform> irradianceGrid : IrradianceGrid;
struct DirectionalCascades {
  viewProj : array<mat4x4<f32>, 4>,
  splits : vec4<f32>,
  params : vec4<f32>, // x = count, y = blend fraction
};
@group(0) @binding(20) var<uniform> directionalCascades : DirectionalCascades;
@group(0) @binding(21) var volumetricFog : texture_3d<f32>;
@group(0) @binding(22) var volumetricFogSampler : sampler;

struct VSOut {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) worldNormal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) worldTangent : vec3<f32>,
  @location(4) tangentSign : f32,
  @location(5) color : vec4<f32>,
  @location(6) @interpolate(linear) previousClip : vec4<f32>,
};

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent : vec4<f32>,
  @location(4) color : vec4<f32>,
};
`;

// StandardMaterial bind group (group 2): factors uniform + texture/sampler pairs.
// Kept separate from HEADER so ShaderMaterial modules can bind their own group 2.
export const MATERIAL_BINDINGS = /* wgsl */ `
struct MaterialU {
  baseColor : vec4<f32>,
  emissive : vec4<f32>,
  params : vec4<f32>,
  misc : vec4<f32>,
  specular : vec4<f32>,  // rgb = specular color factor, w = specular factor
  extra : vec4<f32>,     // x = ior
  sheen : vec4<f32>,     // rgb = sheen color factor, w = sheen roughness
  transmission : vec4<f32>, // x = factor, y = thickness, z = attenuation distance
  attenuation : vec4<f32>,  // rgb = attenuation color
};

@group(2) @binding(0) var<uniform> material : MaterialU;
@group(2) @binding(1) var baseColorTex : texture_2d<f32>;
@group(2) @binding(2) var baseColorSmp : sampler;
@group(2) @binding(3) var normalTex : texture_2d<f32>;
@group(2) @binding(4) var normalSmp : sampler;
@group(2) @binding(5) var mrTex : texture_2d<f32>;
@group(2) @binding(6) var mrSmp : sampler;
@group(2) @binding(7) var emissiveTex : texture_2d<f32>;
@group(2) @binding(8) var emissiveSmp : sampler;
@group(2) @binding(9) var occlusionTex : texture_2d<f32>;
@group(2) @binding(10) var occlusionSmp : sampler;
@group(2) @binding(11) var clearcoatTex : texture_2d<f32>;
@group(2) @binding(12) var clearcoatSmp : sampler;
@group(2) @binding(13) var clearcoatRoughnessTex : texture_2d<f32>;
@group(2) @binding(14) var clearcoatRoughnessSmp : sampler;
`;

// Static vertex stage: transform by the per-object model matrix.
// model.params.x is the shell extrusion distance (world units, 0 for normal
// draws): vertices push out along the world normal to form an inverted hull.
export const VERTEX_STATIC = /* wgsl */ `
struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
  params : vec4<f32>,  // x = shell thickness
  prevModel : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn vs_main(in : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos4 = model.model * vec4<f32>(in.position, 1.0);
  out.worldNormal = normalize((model.normalMat * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldPos = worldPos4.xyz + out.worldNormal * model.params.x;
  out.clipPosition = frame.proj * frame.view * vec4<f32>(out.worldPos, 1.0);
  out.previousClip = frame.prevViewProj * model.prevModel * vec4<f32>(in.position, 1.0);

  out.worldTangent = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Instanced vertex stage: per-instance model matrix from a storage array,
// indexed by instance_index. Normals assume uniform per-instance scale.
export const VERTEX_INSTANCED = /* wgsl */ `
@group(1) @binding(0) var<storage, read> instances : array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read> previousInstances : array<mat4x4<f32>>;

@vertex
fn vs_main(in : VSIn, @builtin(instance_index) ii : u32) -> VSOut {
  var out : VSOut;
  let model = instances[ii];
  let worldPos4 = model * vec4<f32>(in.position, 1.0);
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;
  out.previousClip = frame.prevViewProj * previousInstances[ii] * vec4<f32>(in.position, 1.0);

  out.worldNormal = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldTangent = normalize((model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Skinned vertex stage: blend joint matrices, ignoring the mesh node transform
// (per the glTF skinning spec — joint matrices are already world-space).
export const VERTEX_SKINNED = /* wgsl */ `
@group(3) @binding(0) var<storage, read> bones : array<mat4x4<f32>>;
@group(3) @binding(1) var<storage, read> previousBones : array<mat4x4<f32>>;

// Bound for the shell extrusion distance (params.x); model matrices come from
// the skin (joint matrices are already world-space, per the glTF spec).
struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
  params : vec4<f32>,  // x = shell thickness
  prevModel : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

struct VSInSkinned {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent : vec4<f32>,
  @location(4) joints : vec4<u32>,
  @location(5) weights : vec4<f32>,
  @location(6) color : vec4<f32>,
};

@vertex
fn vs_main(in : VSInSkinned) -> VSOut {
  var out : VSOut;
  let skin =
    in.weights.x * bones[in.joints.x] +
    in.weights.y * bones[in.joints.y] +
    in.weights.z * bones[in.joints.z] +
    in.weights.w * bones[in.joints.w];

  let worldPos4 = skin * vec4<f32>(in.position, 1.0);
  let previousSkin =
    in.weights.x * previousBones[in.joints.x] +
    in.weights.y * previousBones[in.joints.y] +
    in.weights.z * previousBones[in.joints.z] +
    in.weights.w * previousBones[in.joints.w];
  out.worldNormal = normalize((skin * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldPos = worldPos4.xyz + out.worldNormal * model.params.x;
  out.clipPosition = frame.proj * frame.view * vec4<f32>(out.worldPos, 1.0);
  out.previousClip = frame.prevViewProj * previousSkin * vec4<f32>(in.position, 1.0);

  out.worldTangent = normalize((skin * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Morph vertex stage: accumulate weighted POSITION/NORMAL deltas (stored as flat
// f32 arrays indexed [target * vertexCount + vertexId]) onto the base attributes,
// then transform by the per-object model matrix like the static path.
export const VERTEX_MORPH = /* wgsl */ `
struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
  params : vec4<f32>,  // x = shell thickness
  prevModel : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

struct MorphInfo {
  count : u32,
  vertexCount : u32,
  hasNormals : u32,
  _pad : u32,
};
@group(3) @binding(0) var<uniform> morphInfo : MorphInfo;
@group(3) @binding(1) var<storage, read> morphPos : array<f32>;
@group(3) @binding(2) var<storage, read> morphNrm : array<f32>;
@group(3) @binding(3) var<storage, read> morphWeights : array<f32>;
@group(3) @binding(4) var<storage, read> previousMorphWeights : array<f32>;

@vertex
fn vs_main(in : VSIn, @builtin(vertex_index) vid : u32) -> VSOut {
  var out : VSOut;
  var position = in.position;
  var previousPosition = in.position;
  var normal = in.normal;

  for (var t = 0u; t < morphInfo.count; t = t + 1u) {
    let w = morphWeights[t];
    let o = (t * morphInfo.vertexCount + vid) * 3u;
    position = position + w * vec3<f32>(morphPos[o], morphPos[o + 1u], morphPos[o + 2u]);
    previousPosition = previousPosition + previousMorphWeights[t] * vec3<f32>(morphPos[o], morphPos[o + 1u], morphPos[o + 2u]);
    if (morphInfo.hasNormals != 0u) {
      normal = normal + w * vec3<f32>(morphNrm[o], morphNrm[o + 1u], morphNrm[o + 2u]);
    }
  }

  let worldPos4 = model.model * vec4<f32>(position, 1.0);
  out.worldNormal = normalize((model.normalMat * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos4.xyz + out.worldNormal * model.params.x;
  out.clipPosition = frame.proj * frame.view * vec4<f32>(out.worldPos, 1.0);
  out.previousClip = frame.prevViewProj * model.prevModel * vec4<f32>(previousPosition, 1.0);

  out.worldTangent = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Shared shading helpers: BRDF lobes, tonemap/sRGB encode, env + shadow
// sampling, clustered light lookup, and per-light evaluation. Exported so
// generated ShaderMaterial fragments reuse the engine lighting verbatim.
export const SHADE_HELPERS = /* wgsl */ `
fn distributionGGX(NoH : f32, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / max(PI * d * d, 1e-7);
}

fn visibilitySmith(NoV : f32, NoL : f32, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let ggxV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let ggxL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(ggxV + ggxL, 1e-7);
}

fn fresnelSchlick(VoH : f32, f0 : vec3<f32>) -> vec3<f32> {
  let f = pow(clamp(1.0 - VoH, 0.0, 1.0), 5.0);
  return f0 + (vec3<f32>(1.0) - f0) * f;
}

// Charlie sheen NDF + Ashikhmin visibility (KHR_materials_sheen), for cloth-like
// retroreflection at grazing angles.
fn distributionCharlie(NoH : f32, roughness : f32) -> f32 {
  let invR = 1.0 / roughness;
  let cos2h = NoH * NoH;
  let sin2h = max(1.0 - cos2h, 0.0078125);
  return (2.0 + invR) * pow(sin2h, invR * 0.5) / (2.0 * PI);
}

fn visibilitySheen(NoV : f32, NoL : f32) -> f32 {
  return 1.0 / max(4.0 * (NoL + NoV - NoL * NoV), 1e-7);
}

fn acesFilmic(x : vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Equirectangular lookup: direction -> panorama UV, sampled at an explicit mip.
fn dirToEquirectUv(d : vec3<f32>) -> vec2<f32> {
  let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2<f32>(u, v);
}

fn sampleEnv(dir : vec3<f32>, lod : f32) -> vec3<f32> {
  return textureSampleLevel(envMap, envSampler, dirToEquirectUv(normalize(dir)), lod).rgb;
}

// Karis' analytic environment-BRDF fit (avoids a precomputed LUT).
fn envBRDFApprox(roughness : f32, NoV : f32) -> vec2<f32> {
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn sampleShadowCascade(cascade : u32, worldPos : vec3<f32>, N : vec3<f32>) -> f32 {
  let bias = frame.shadowParams.z;
  let lp = directionalCascades.viewProj[cascade] * vec4<f32>(worldPos + N * bias, 1.0);
  let ndc = lp.xyz / lp.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  var uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
  let count = u32(directionalCascades.params.x);
  let scale = select(1.0, 0.5, count > 1u);
  let tileOffset = vec2<f32>(f32(cascade % 2u), f32(cascade / 2u)) * scale;
  uv = tileOffset + uv * scale;
  let texel = 1.0 / frame.shadowParams.y;
  let tileMin = tileOffset + vec2<f32>(texel * 1.5);
  let tileMax = tileOffset + vec2<f32>(scale - texel * 1.5);
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
      sum = sum + textureSampleCompareLevel(shadowMap, shadowSampler, clamp(uv + offset, tileMin, tileMax), ndc.z);
    }
  }
  return sum / 9.0;
}

// 3x3 PCF against the directional atlas, blending across cascade seams.
fn sampleShadow(worldPos : vec3<f32>, N : vec3<f32>) -> f32 {
  if (frame.shadowParams.x < 0.5) { return 1.0; }
  let count = u32(directionalCascades.params.x);
  let viewDepth = -(frame.view * vec4<f32>(worldPos, 1.0)).z;
  var cascade = 0u;
  while (cascade + 1u < count && viewDepth > directionalCascades.splits[cascade]) {
    cascade = cascade + 1u;
  }
  let current = sampleShadowCascade(cascade, worldPos, N);
  if (cascade + 1u >= count) { return current; }
  var start = 0.0;
  if (cascade > 0u) { start = directionalCascades.splits[cascade - 1u]; }
  let width = (directionalCascades.splits[cascade] - start) * directionalCascades.params.y;
  let blend = smoothstep(directionalCascades.splits[cascade] - width, directionalCascades.splits[cascade], viewDepth);
  if (blend <= 0.0) { return current; }
  return mix(current, sampleShadowCascade(cascade + 1u, worldPos, N), blend);
}

// 3x3 PCF against the spot-shadow atlas for one tile.
fn sampleSpotShadow(tileIdx : u32, worldPos : vec3<f32>, N : vec3<f32>, bias : f32) -> f32 {
  let tile = shadowTiles[tileIdx];
  let lp = tile.viewProj * vec4<f32>(worldPos + N * bias, 1.0);
  if (lp.w <= 0.0) { return 1.0; }
  let ndc = lp.xyz / lp.w;
  if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
  let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  let atlasUV = tile.region.xy + uv * tile.region.z;
  let step = tile.region.w;
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let off = vec2<f32>(f32(dx), f32(dy)) * step;
      sum = sum + textureSampleCompareLevel(spotAtlas, spotAtlasCmp, atlasUV + off, ndc.z);
    }
  }
  return sum / 9.0;
}

fn linearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, cutoff);
}

// Clustered forward+ light list for a fragment. Returns x = light count and
// y = the cluster's base offset into clusterLights (light indices start at
// y + 1); y == 0xffffffffu means clustering is off and j indexes lights directly.
fn lightList(clipXY : vec2<f32>, worldPos : vec3<f32>) -> vec2<u32> {
  if (frame.clusterParams.x < 0.5) {
    return vec2<u32>(u32(frame.cameraPos.w), 0xffffffffu);
  }
  let tx = min(u32(clipXY.x / frame.clusterDims.x), CLUSTER_X - 1u);
  let ty = min(u32(clipXY.y / frame.clusterDims.y), CLUSTER_Y - 1u);
  let near = frame.clusterParams.y;
  let far = frame.clusterParams.z;
  let viewZ = max(-(frame.view * vec4<f32>(worldPos, 1.0)).z, near);
  let slice = min(u32(log(viewZ / near) / log(far / near) * f32(CLUSTER_Z)), CLUSTER_Z - 1u);
  let base = (tx + ty * CLUSTER_X + slice * CLUSTER_X * CLUSTER_Y) * (MAX_PER_CLUSTER + 1u);
  return vec2<u32>(min(clusterLights[base], MAX_PER_CLUSTER), base);
}

fn lightIndex(list : vec2<u32>, j : u32) -> u32 {
  if (list.y == 0xffffffffu) { return j; }
  return clusterLights[list.y + 1u + j];
}

// Direction + shadowed/attenuated radiance of packed light i at a surface point.
struct LightContrib {
  L : vec3<f32>,
  radiance : vec3<f32>,
};

fn evaluateLight(i : u32, worldPos : vec3<f32>, N : vec3<f32>) -> LightContrib {
  let light = lights[i];
  let kind = u32(light.positionKind.w);

  var out : LightContrib;
  var attenuation = 1.0;
  var radiance = light.colorDecay.xyz;

  if (kind == LIGHT_POINT) {
    let toLight = light.positionKind.xyz - worldPos;
    let dist = length(toLight);
    out.L = toLight / max(dist, 1e-4);
    let decay = light.colorDecay.w;
    attenuation = 1.0 / max(pow(dist, decay), 1e-4);
    let range = light.directionRange.w;
    if (range > 0.0) {
      let f = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
      attenuation = attenuation * f * f;
    }
    // Cube-face shadow: spotParams.z holds the first of 6 consecutive atlas
    // tiles (+X,-X,+Y,-Y,+Z,-Z); pick the face on the dominant axis of the
    // light→fragment direction.
    let tileIdx = i32(light.spotParams.z);
    if (tileIdx >= 0) {
      let d = -toLight;
      let ad = abs(d);
      var face = 0u;
      if (ad.x >= ad.y && ad.x >= ad.z) {
        face = select(1u, 0u, d.x > 0.0);
      } else if (ad.y >= ad.z) {
        face = select(3u, 2u, d.y > 0.0);
      } else {
        face = select(5u, 4u, d.z > 0.0);
      }
      radiance = radiance * sampleSpotShadow(u32(tileIdx) + face, worldPos, N, frame.shadowParams.z);
    }
  } else if (kind == LIGHT_SPOT) {
    let toLight = light.positionKind.xyz - worldPos;
    let dist = length(toLight);
    out.L = toLight / max(dist, 1e-4);
    let decay = light.colorDecay.w;
    attenuation = 1.0 / max(pow(dist, decay), 1e-4);
    let range = light.directionRange.w;
    if (range > 0.0) {
      let f = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
      attenuation = attenuation * f * f;
    }
    // Angular falloff between inner and outer cone.
    let cosTheta = dot(-out.L, normalize(light.directionRange.xyz));
    let angleFactor = clamp(
      (cosTheta - light.spotParams.y) / max(light.spotParams.x - light.spotParams.y, 1e-4),
      0.0, 1.0);
    attenuation = attenuation * angleFactor * angleFactor;
    // Spot shadow atlas.
    let tileIdx = i32(light.spotParams.z);
    if (tileIdx >= 0) {
      radiance = radiance * sampleSpotShadow(u32(tileIdx), worldPos, N, frame.shadowParams.z);
    }
  } else {
    out.L = -light.directionRange.xyz;
    // The designated directional caster is attenuated by the shadow map.
    if (i == u32(frame.shadowParams.w)) {
      radiance = radiance * sampleShadow(worldPos, N);
    }
  }

  out.radiance = radiance * attenuation;
  return out;
}

// Distance fog: mix toward fogColor by view distance (linear or exp2 mode),
// optionally thinning with the fragment's altitude. No-op when mode is 0.
fn applyFog(color : vec3<f32>, worldPos : vec3<f32>) -> vec3<f32> {
  let mode = u32(frame.fogColor.w);
  if (mode == 0u) { return color; }
  if (frame.fogParams.w > 0.5) {
    let clip = frame.proj * frame.view * vec4<f32>(worldPos, 1.0);
    let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let viewDepth = -(frame.view * vec4<f32>(worldPos, 1.0)).z;
    let z = clamp((viewDepth - frame.clusterParams.y) / max(frame.clusterParams.z - frame.clusterParams.y, 1e-4), 0.0, 1.0);
    let fog = textureSampleLevel(volumetricFog, volumetricFogSampler, vec3<f32>(uv, z), 0.0);
    return color * fog.a + fog.rgb;
  }
  let dist = distance(worldPos, frame.cameraPos.xyz);
  var amount : f32;
  if (mode == 1u) {
    amount = clamp((dist - frame.fogParams.x) / max(frame.fogParams.y - frame.fogParams.x, 1e-4), 0.0, 1.0);
  } else {
    let dd = frame.fogParams.x * dist;
    amount = 1.0 - exp(-dd * dd);
  }
  let falloff = frame.fogParams.z;
  if (falloff > 0.0) {
    amount = amount * exp(-falloff * max(worldPos.y, 0.0));
  }
  return mix(color, frame.fogColor.rgb, clamp(amount, 0.0, 1.0));
}

// Indirect light: image-based when an environment is bound, flat ambient otherwise.
fn probeIntensity(index : u32) -> f32 {
  return reflectionProbes.intensity[index];
}

fn localReflection(worldPos : vec3<f32>, R : vec3<f32>, lod : f32) -> vec4<f32> {
  let count = u32(reflectionProbes.info.x);
  var first = 1e30;
  var second = 1e30;
  var firstIndex = 0u;
  var secondIndex = 0u;
  for (var i = 0u; i < count; i = i + 1u) {
    let p = reflectionProbes.positionRadius[i];
    let d = distance(worldPos, p.xyz) / max(p.w, 1e-4);
    if (d < first) {
      second = first; secondIndex = firstIndex;
      first = d; firstIndex = i;
    } else if (d < second) {
      second = d; secondIndex = i;
    }
  }
  if (count == 0u || first >= 1.0) { return vec4<f32>(0.0); }
  let uv = dirToEquirectUv(R);
  let a = textureSampleLevel(reflectionMaps, reflectionSampler, uv, i32(firstIndex), lod).rgb * probeIntensity(firstIndex);
  if (count == 1u || second >= 1.0) { return vec4<f32>(a, 1.0 - smoothstep(0.8, 1.0, first)); }
  let b = textureSampleLevel(reflectionMaps, reflectionSampler, uv, i32(secondIndex), lod).rgb * probeIntensity(secondIndex);
  let wa = 1.0 / max(first, 1e-3);
  let wb = 1.0 / max(second, 1e-3);
  return vec4<f32>((a * wa + b * wb) / (wa + wb), 1.0 - smoothstep(0.8, 1.0, first));
}

fn shBasis(i : u32, d : vec3<f32>) -> f32 {
  if (i == 0u) { return 0.282095; }
  if (i == 1u) { return 0.488603 * d.y; }
  if (i == 2u) { return 0.488603 * d.z; }
  if (i == 3u) { return 0.488603 * d.x; }
  if (i == 4u) { return 1.092548 * d.x * d.y; }
  if (i == 5u) { return 1.092548 * d.y * d.z; }
  if (i == 6u) { return 0.315392 * (3.0 * d.z * d.z - 1.0); }
  if (i == 7u) { return 1.092548 * d.x * d.z; }
  return 0.546274 * (d.x * d.x - d.y * d.y);
}

fn probeIrradiance(index : u32, N : vec3<f32>) -> vec3<f32> {
  var value = vec3<f32>(0.0);
  for (var i = 0u; i < 9u; i = i + 1u) {
    value = value + irradianceCoefficients[index * 9u + i].rgb * shBasis(i, N);
  }
  return max(value, vec3<f32>(0.0));
}

fn sampleIrradianceGrid(worldPos : vec3<f32>, N : vec3<f32>) -> vec4<f32> {
  if (irradianceGrid.origin.w < 0.5) { return vec4<f32>(0.0); }
  let dims = vec3<u32>(irradianceGrid.dims.xyz);
  let coord = (worldPos - irradianceGrid.origin.xyz) / irradianceGrid.spacing.xyz;
  let maxCoord = vec3<f32>(dims - vec3<u32>(1u));
  if (any(coord < vec3<f32>(0.0)) || any(coord > maxCoord)) { return vec4<f32>(0.0); }
  let c0 = vec3<u32>(floor(coord));
  let c1 = min(c0 + vec3<u32>(1u), dims - vec3<u32>(1u));
  let f = fract(coord);
  let nx = dims.x;
  let ny = dims.y;
  let i000 = c0.x + nx * (c0.y + ny * c0.z);
  let i100 = c1.x + nx * (c0.y + ny * c0.z);
  let i010 = c0.x + nx * (c1.y + ny * c0.z);
  let i110 = c1.x + nx * (c1.y + ny * c0.z);
  let i001 = c0.x + nx * (c0.y + ny * c1.z);
  let i101 = c1.x + nx * (c0.y + ny * c1.z);
  let i011 = c0.x + nx * (c1.y + ny * c1.z);
  let i111 = c1.x + nx * (c1.y + ny * c1.z);
  let z0 = mix(mix(probeIrradiance(i000, N), probeIrradiance(i100, N), f.x),
               mix(probeIrradiance(i010, N), probeIrradiance(i110, N), f.x), f.y);
  let z1 = mix(mix(probeIrradiance(i001, N), probeIrradiance(i101, N), f.x),
               mix(probeIrradiance(i011, N), probeIrradiance(i111, N), f.x), f.y);
  return vec4<f32>(mix(z0, z1, f.z), 1.0);
}

fn indirectLight(worldPos : vec3<f32>, N : vec3<f32>, V : vec3<f32>, NoV : f32, roughness : f32,
                 f0 : vec3<f32>, diffuseColor : vec3<f32>, ao : f32) -> vec3<f32> {
  let R = reflect(-V, N);
  let probe = localReflection(worldPos, R, roughness * 5.0);
  let grid = sampleIrradianceGrid(worldPos, N);
  if (frame.envParams.x > 0.5) {
    let maxMip = frame.envParams.z;
    var prefiltered = sampleEnv(R, roughness * maxMip); // specular (raw mip or IBL prefiltered)
    prefiltered = mix(prefiltered, probe.rgb, probe.a);
    let useIBL = (u32(frame.envParams.w) & 2u) != 0u;
    var diffuseIBL : vec3<f32>;
    var ab          : vec2<f32>;
    if (useIBL) {
      // True GGX IBL: cosine-convolved irradiance map + split-sum BRDF LUT.
      diffuseIBL = textureSampleLevel(irrMap, irrSampler, dirToEquirectUv(N), 0.0).rgb;
      ab = textureSampleLevel(brdfLUT, brdfSampler, clamp(vec2<f32>(NoV, roughness), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rg;
    } else {
      // Fallback: mip-chain approximation + Karis analytic BRDF.
      diffuseIBL = sampleEnv(N, maxMip);
      ab = envBRDFApprox(roughness, NoV);
    }
    diffuseIBL = mix(diffuseIBL, grid.rgb, grid.a);
    let specularIBL = prefiltered * (f0 * ab.x + ab.y);
    return (diffuseIBL * diffuseColor + specularIBL) * ao * frame.envParams.y;
  }
  let ab = textureSampleLevel(brdfLUT, brdfSampler, clamp(vec2<f32>(NoV, roughness), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rg;
  let localSpecular = probe.rgb * (f0 * ab.x + ab.y) * probe.a;
  let diffuseIrradiance = mix(frame.ambient.rgb, grid.rgb, grid.a);
  return (diffuseIrradiance * diffuseColor + localSpecular) * ao;
}

`;

// PBR fragment stage: decode StandardMaterial inputs, shade, tonemap.
const FRAGMENT = SHADE_HELPERS + /* wgsl */ `
struct ShadedSurface {
  color : vec4<f32>,
  normal : vec3<f32>,
};

// Shared shading: returns linear HDR color (before exposure/tonemap), alpha, and shading normal.
fn shadeSurface(in : VSOut, frontFacing : bool) -> ShadedSurface {
  let flags = u32(material.misc.y);
  let hasNormalMap = (flags & 2u) != 0u;

  let baseSample = textureSample(baseColorTex, baseColorSmp, in.uv);
  let baseColor = material.baseColor.rgb * baseSample.rgb * in.color.rgb;
  let alpha = material.baseColor.a * baseSample.a * in.color.a;

  let alphaCutoff = material.misc.x;
  if (alphaCutoff > 0.0 && alpha < alphaCutoff) {
    discard;
  }

  let mrSample = textureSample(mrTex, mrSmp, in.uv);
  let metalness = clamp(material.params.x * mrSample.b, 0.0, 1.0);
  var roughness = clamp(material.params.y * mrSample.g, 0.04, 1.0);

  // Clear-coat: a thin dielectric specular layer over the base (KHR_materials_clearcoat).
  // White default textures → factor * 1.0 = factor (backward-compatible).
  let clearcoat = clamp(material.misc.z * textureSample(clearcoatTex, clearcoatSmp, in.uv).r, 0.0, 1.0);
  let clearcoatRoughness = clamp(material.misc.w * textureSample(clearcoatRoughnessTex, clearcoatRoughnessSmp, in.uv).g, 0.04, 1.0);

  // Sheen: a soft retroreflective lobe for cloth (KHR_materials_sheen).
  let sheenColor = material.sheen.rgb;
  let sheenRoughness = clamp(material.sheen.w, 0.07, 1.0);
  let hasSheen = sheenColor.r + sheenColor.g + sheenColor.b > 0.0;

  var N = normalize(in.worldNormal);
  if (!frontFacing) {
    N = -N;
  }
  if (hasNormalMap) {
    var T = normalize(in.worldTangent - N * dot(in.worldTangent, N));
    let B = cross(N, T) * in.tangentSign;
    var tn = textureSample(normalTex, normalSmp, in.uv).xyz * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * material.params.z, tn.z);
    N = normalize(mat3x3<f32>(T, B, N) * tn);
  }

  let V = normalize(frame.cameraPos.xyz - in.worldPos);
  let NoV = max(dot(N, V), 1e-4);

  // Dielectric F0 from IOR (KHR_materials_ior), tinted/scaled by specular
  // (KHR_materials_specular). Defaults (ior 1.5, white, factor 1) give 0.04.
  let ior = material.extra.x;
  let iorF0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
  let dielectricF0 = min(vec3<f32>(iorF0) * material.specular.rgb, vec3<f32>(1.0)) * material.specular.w;
  let f0 = mix(dielectricF0, baseColor, metalness);
  let diffuseColor = baseColor * (1.0 - metalness);

  var color = vec3<f32>(0.0);

  // Direct light: clustered forward+ when enabled, all lights otherwise.
  let list = lightList(in.clipPosition.xy, in.worldPos);
  for (var j = 0u; j < list.x; j = j + 1u) {
    let i = lightIndex(list, j);
    let lc = evaluateLight(i, in.worldPos, N);
    let L = lc.L;

    let H = normalize(V + L);
    let NoL = max(dot(N, L), 0.0);
    if (NoL <= 0.0) { continue; }
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    let D = distributionGGX(NoH, roughness);
    let Vis = visibilitySmith(NoV, NoL, roughness);
    let F = fresnelSchlick(VoH, f0);

    let specular = D * Vis * F;
    let kd = (vec3<f32>(1.0) - F);
    let diffuse = kd * diffuseColor / PI;
    var lit = diffuse + specular;

    // Add a sheen lobe (Charlie NDF) over the base layer.
    if (hasSheen) {
      let Ds = distributionCharlie(NoH, sheenRoughness);
      let Vs = visibilitySheen(NoV, NoL);
      lit = lit + sheenColor * (Ds * Vs);
    }

    // Add a clear-coat GGX lobe and attenuate the base layer by its Fresnel.
    if (clearcoat > 0.0) {
      let Dc = distributionGGX(NoH, clearcoatRoughness);
      let Vc = visibilitySmith(NoV, NoL, clearcoatRoughness);
      let Fc = fresnelSchlick(VoH, vec3<f32>(0.04)).x * clearcoat;
      lit = lit * (1.0 - Fc) + vec3<f32>(Dc * Vc * Fc);
    }

    color = color + lit * lc.radiance * NoL;
  }

  let aoSample = textureSample(occlusionTex, occlusionSmp, in.uv).r;
  let ao = mix(1.0, aoSample, material.params.w);

  color = color + indirectLight(in.worldPos, N, V, NoV, roughness, f0, diffuseColor, ao);

  let emissiveSample = textureSample(emissiveTex, emissiveSmp, in.uv).rgb;
  color = color + material.emissive.rgb * material.emissive.a * emissiveSample;

  // Transmission: refract through the surface, attenuate by Beer-Lambert volume.
  // When screen-space scene sampling is active (bit 2 of envParams.w), sample the opaque scene capture at
  // a screen-space UV perturbed by the refraction direction; otherwise fall back to
  // the environment map.
  let transmissionFactor = material.transmission.x;
  if (transmissionFactor > 0.0) {
    let refr = refract(-V, N, 1.0 / max(ior, 1.0001));
    var background = frame.ambient.rgb;
    let useSceneCapture = (u32(frame.envParams.w) & 4u) != 0u;
    if (useSceneCapture) {
      // Project (worldPos + refr * thickness) to screen UV and sample the opaque snapshot.
      let thickness = max(material.transmission.y, 0.05);
      let refractedWorld = in.worldPos + refr * thickness;
      let clipRefr = frame.proj * frame.view * vec4<f32>(refractedWorld, 1.0);
      let ndcRefr = clipRefr.xy / clipRefr.w;
      let ssrUV = clamp(ndcRefr * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
      background = textureSampleLevel(sceneCapture, envSampler, ssrUV, 0.0).rgb;
    } else if (frame.envParams.x > 0.5) {
      background = sampleEnv(refr, roughness * frame.envParams.z) * frame.envParams.y;
    }
    var attenuation = vec3<f32>(1.0);
    let attenuationDistance = material.transmission.z;
    if (attenuationDistance > 0.0) {
      let absorbance = -log(clamp(material.attenuation.rgb, vec3<f32>(1e-4), vec3<f32>(1.0))) / attenuationDistance;
      attenuation = exp(-absorbance * material.transmission.y);
    }
    let transmitted = background * attenuation * baseColor;
    color = mix(color, transmitted, transmissionFactor);
  }

  color = applyFog(color, in.worldPos);
  var out : ShadedSurface;
  out.color = vec4<f32>(color, alpha);
  out.normal = N;
  return out;
}

@fragment
fn fs_main(in : VSOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  let s = shadeSurface(in, frontFacing);
  var color = s.color.rgb * frame.ambient.w; // exposure
  // envParams.w bit 0: linear output (post pipeline tonemaps later).
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    color = acesFilmic(color);
    color = linearToSRGB(color);
  }
  return vec4<f32>(color, s.color.a);
}

struct SceneOut {
  @location(0) color : vec4<f32>,
  @location(1) normalDepth : vec4<f32>,
  @location(2) velocity : vec2<f32>,
};

@fragment
fn fs_scene(in : VSOut, @builtin(front_facing) frontFacing : bool) -> SceneOut {
  let s = shadeSurface(in, frontFacing);
  var color = s.color.rgb * frame.ambient.w;
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    color = linearToSRGB(acesFilmic(color));
  }
  let viewDepth = -(frame.view * vec4<f32>(in.worldPos, 1.0)).z;
  let linearDepth = clamp(
    (viewDepth - frame.clusterParams.y) / max(frame.clusterParams.z - frame.clusterParams.y, 1e-6),
    0.0, 1.0,
  );
  var out : SceneOut;
  out.color = vec4<f32>(color, s.color.a);
  out.normalDepth = vec4<f32>(s.normal, linearDepth);
  let previousNDC = in.previousClip.xy / max(in.previousClip.w, 1e-6);
  let currentUV = in.clipPosition.xy / vec2<f32>(frame.clusterDims.x * f32(CLUSTER_X), frame.clusterDims.y * f32(CLUSTER_Y));
  let previousUV = previousNDC * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  out.velocity = currentUV - previousUV;
  return out;
}

// Weighted-blended OIT (McGuire/Bavoil). Accumulates premultiplied linear color
// weighted by depth+alpha into target 0, and product of (1-a) into target 1.
struct OITOut {
  @location(0) accum : vec4<f32>,
  @location(1) reveal : f32,
};

@fragment
fn fs_oit(in : VSOut, @builtin(front_facing) frontFacing : bool) -> OITOut {
  let s = shadeSurface(in, frontFacing);
  let color = s.color.rgb * frame.ambient.w; // exposure
  let a = clamp(s.color.a, 0.0, 1.0);
  let viewZ = abs((frame.view * vec4<f32>(in.worldPos, 1.0)).z);
  let weight = a * clamp(0.03 / (1e-5 + pow(viewZ / 200.0, 4.0)), 1e-2, 3e3);
  var out : OITOut;
  out.accum = vec4<f32>(color * a, a) * weight;
  out.reveal = a;
  return out;
}
`;

export const PBR_SHADER = HEADER + MATERIAL_BINDINGS + VERTEX_STATIC + FRAGMENT;
export const PBR_SKINNED_SHADER = HEADER + MATERIAL_BINDINGS + VERTEX_SKINNED + FRAGMENT;
export const PBR_INSTANCED_SHADER = HEADER + MATERIAL_BINDINGS + VERTEX_INSTANCED + FRAGMENT;
export const PBR_MORPH_SHADER = HEADER + MATERIAL_BINDINGS + VERTEX_MORPH + FRAGMENT;
