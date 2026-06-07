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

// Shared: constants, uniform/storage layout, varyings.
const HEADER = /* wgsl */ `
const PI = 3.141592653589793;
const LIGHT_DIRECTIONAL = 0u;
const LIGHT_POINT = 1u;

struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  cameraPos : vec4<f32>,   // xyz = position, w = numLights
  ambient : vec4<f32>,     // rgb = ambient irradiance, w = exposure
};

struct Light {
  positionKind : vec4<f32>,
  directionRange : vec4<f32>,
  colorDecay : vec4<f32>,
};

struct MaterialU {
  baseColor : vec4<f32>,
  emissive : vec4<f32>,
  params : vec4<f32>,
  misc : vec4<f32>,
  specular : vec4<f32>,  // rgb = specular color factor, w = specular factor
  extra : vec4<f32>,     // x = ior
  sheen : vec4<f32>,     // rgb = sheen color factor, w = sheen roughness
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var<storage, read> lights : array<Light>;

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

struct VSOut {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) worldNormal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) worldTangent : vec3<f32>,
  @location(4) tangentSign : f32,
  @location(5) color : vec4<f32>,
};

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent : vec4<f32>,
  @location(4) color : vec4<f32>,
};
`;

// Static vertex stage: transform by the per-object model matrix.
const VERTEX_STATIC = /* wgsl */ `
struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn vs_main(in : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos4 = model.model * vec4<f32>(in.position, 1.0);
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;

  out.worldNormal = normalize((model.normalMat * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldTangent = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Instanced vertex stage: per-instance model matrix from a storage array,
// indexed by instance_index. Normals assume uniform per-instance scale.
const VERTEX_INSTANCED = /* wgsl */ `
@group(1) @binding(0) var<storage, read> instances : array<mat4x4<f32>>;

@vertex
fn vs_main(in : VSIn, @builtin(instance_index) ii : u32) -> VSOut {
  var out : VSOut;
  let model = instances[ii];
  let worldPos4 = model * vec4<f32>(in.position, 1.0);
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;

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
const VERTEX_SKINNED = /* wgsl */ `
@group(3) @binding(0) var<storage, read> bones : array<mat4x4<f32>>;

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
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;

  out.worldNormal = normalize((skin * vec4<f32>(in.normal, 0.0)).xyz);
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
const VERTEX_MORPH = /* wgsl */ `
struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
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

@vertex
fn vs_main(in : VSIn, @builtin(vertex_index) vid : u32) -> VSOut {
  var out : VSOut;
  var position = in.position;
  var normal = in.normal;

  for (var t = 0u; t < morphInfo.count; t = t + 1u) {
    let w = morphWeights[t];
    let o = (t * morphInfo.vertexCount + vid) * 3u;
    position = position + w * vec3<f32>(morphPos[o], morphPos[o + 1u], morphPos[o + 2u]);
    if (morphInfo.hasNormals != 0u) {
      normal = normal + w * vec3<f32>(morphNrm[o], morphNrm[o + 1u], morphNrm[o + 2u]);
    }
  }

  let worldPos4 = model.model * vec4<f32>(position, 1.0);
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;

  out.worldNormal = normalize((model.normalMat * vec4<f32>(normal, 0.0)).xyz);
  out.worldTangent = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  out.color = in.color;
  return out;
}
`;

// Shared fragment stage: PBR shading, tonemap, sRGB encode.
const FRAGMENT = /* wgsl */ `
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

fn linearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, cutoff);
}

@fragment
fn fs_main(in : VSOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
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
  let clearcoat = clamp(material.misc.z, 0.0, 1.0);
  let clearcoatRoughness = clamp(material.misc.w, 0.04, 1.0);

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
  let numLights = u32(frame.cameraPos.w);

  for (var i = 0u; i < numLights; i = i + 1u) {
    let light = lights[i];
    let kind = u32(light.positionKind.w);

    var L : vec3<f32>;
    var attenuation = 1.0;
    var radiance = light.colorDecay.xyz;

    if (kind == LIGHT_POINT) {
      let toLight = light.positionKind.xyz - in.worldPos;
      let dist = length(toLight);
      L = toLight / max(dist, 1e-4);
      let decay = light.colorDecay.w;
      attenuation = 1.0 / max(pow(dist, decay), 1e-4);
      let range = light.directionRange.w;
      if (range > 0.0) {
        let f = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
        attenuation = attenuation * f * f;
      }
    } else {
      L = -light.directionRange.xyz;
    }

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

    radiance = radiance * attenuation * NoL;
    color = color + lit * radiance;
  }

  let aoSample = textureSample(occlusionTex, occlusionSmp, in.uv).r;
  let ao = mix(1.0, aoSample, material.params.w);
  color = color + frame.ambient.rgb * diffuseColor * ao;

  let emissiveSample = textureSample(emissiveTex, emissiveSmp, in.uv).rgb;
  color = color + material.emissive.rgb * material.emissive.a * emissiveSample;

  color = color * frame.ambient.w;
  color = acesFilmic(color);
  color = linearToSRGB(color);

  return vec4<f32>(color, alpha);
}
`;

export const PBR_SHADER = HEADER + VERTEX_STATIC + FRAGMENT;
export const PBR_SKINNED_SHADER = HEADER + VERTEX_SKINNED + FRAGMENT;
export const PBR_INSTANCED_SHADER = HEADER + VERTEX_INSTANCED + FRAGMENT;
export const PBR_MORPH_SHADER = HEADER + VERTEX_MORPH + FRAGMENT;
