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
const LIGHT_SPOT = 2u;

struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  cameraPos : vec4<f32>,   // xyz = position, w = numLights
  ambient : vec4<f32>,     // rgb = ambient irradiance, w = exposure
  lightViewProj : mat4x4<f32>, // directional shadow caster's view-projection
  shadowParams : vec4<f32>,    // x = enabled, y = map size, z = normal bias, w = caster light index
  envParams : vec4<f32>,       // x = enabled, y = intensity, z = max mip level, w = flags (bit0=linearOut,bit1=IBL)
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
@group(0) @binding(13) var sceneCapture : texture_2d<f32>; // opaque HDR snapshot for SSR

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

// 3x3 PCF against the directional shadow atlas. Returns 1 (fully lit) when
// shadows are disabled or the point falls outside the light frustum.
fn sampleShadow(worldPos : vec3<f32>, N : vec3<f32>) -> f32 {
  if (frame.shadowParams.x < 0.5) { return 1.0; }
  let bias = frame.shadowParams.z;
  let lp = frame.lightViewProj * vec4<f32>(worldPos + N * bias, 1.0);
  let ndc = lp.xyz / lp.w;
  if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
    return 1.0;
  }
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
  let texel = 1.0 / frame.shadowParams.y;
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
      sum = sum + textureSampleCompareLevel(shadowMap, shadowSampler, uv + offset, ndc.z);
    }
  }
  return sum / 9.0;
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

// Shared shading: returns linear HDR color (before exposure/tonemap) + alpha.
fn shadeSurface(in : VSOut, frontFacing : bool) -> vec4<f32> {
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
        radiance = radiance * sampleSpotShadow(u32(tileIdx) + face, in.worldPos, N, frame.shadowParams.z);
      }
    } else if (kind == LIGHT_SPOT) {
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
      // Angular falloff between inner and outer cone.
      let cosTheta = dot(-L, normalize(light.directionRange.xyz));
      let angleFactor = clamp(
        (cosTheta - light.spotParams.y) / max(light.spotParams.x - light.spotParams.y, 1e-4),
        0.0, 1.0);
      attenuation = attenuation * angleFactor * angleFactor;
      // Spot shadow atlas.
      let tileIdx = i32(light.spotParams.z);
      if (tileIdx >= 0) {
        radiance = radiance * sampleSpotShadow(u32(tileIdx), in.worldPos, N, frame.shadowParams.z);
      }
    } else {
      L = -light.directionRange.xyz;
      // The designated directional caster is attenuated by the shadow map.
      if (i == u32(frame.shadowParams.w)) {
        radiance = radiance * sampleShadow(in.worldPos, N);
      }
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

  // Indirect light: image-based when an environment is bound, else flat ambient.
  if (frame.envParams.x > 0.5) {
    let maxMip = frame.envParams.z;
    let R = reflect(-V, N);
    let prefiltered = sampleEnv(R, roughness * maxMip); // specular (raw mip or IBL prefiltered)
    let useIBL = (u32(frame.envParams.w) & 2u) != 0u;
    var diffuseIBL : vec3<f32>;
    var ab          : vec2<f32>;
    if (useIBL) {
      // True GGX IBL: cosine-convolved irradiance map + split-sum BRDF LUT.
      diffuseIBL = textureSampleLevel(irrMap, irrSampler, dirToEquirectUv(N), 0.0).rgb;
      ab = textureSample(brdfLUT, brdfSampler, clamp(vec2<f32>(NoV, roughness), vec2<f32>(0.0), vec2<f32>(1.0))).rg;
    } else {
      // Fallback: mip-chain approximation + Karis analytic BRDF.
      diffuseIBL = sampleEnv(N, maxMip);
      ab = envBRDFApprox(roughness, NoV);
    }
    let specularIBL = prefiltered * (f0 * ab.x + ab.y);
    color = color + (diffuseIBL * diffuseColor + specularIBL) * ao * frame.envParams.y;
  } else {
    color = color + frame.ambient.rgb * diffuseColor * ao;
  }

  let emissiveSample = textureSample(emissiveTex, emissiveSmp, in.uv).rgb;
  color = color + material.emissive.rgb * material.emissive.a * emissiveSample;

  // Transmission: refract through the surface, attenuate by Beer-Lambert volume.
  // When SSR is active (bit 2 of envParams.w), sample the opaque scene capture at
  // a screen-space UV perturbed by the refraction direction; otherwise fall back to
  // the environment map.
  let transmissionFactor = material.transmission.x;
  if (transmissionFactor > 0.0) {
    let refr = refract(-V, N, 1.0 / max(ior, 1.0001));
    var background = frame.ambient.rgb;
    let useSSR = (u32(frame.envParams.w) & 4u) != 0u;
    if (useSSR) {
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

  return vec4<f32>(color, alpha);
}

@fragment
fn fs_main(in : VSOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  let s = shadeSurface(in, frontFacing);
  var color = s.rgb * frame.ambient.w; // exposure
  // envParams.w bit 0: linear output (post pipeline tonemaps later).
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    color = acesFilmic(color);
    color = linearToSRGB(color);
  }
  return vec4<f32>(color, s.a);
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
  let color = s.rgb * frame.ambient.w; // exposure (stays linear; tonemapped after composite)
  let a = clamp(s.a, 0.0, 1.0);
  let viewZ = abs((frame.view * vec4<f32>(in.worldPos, 1.0)).z);
  let weight = a * clamp(0.03 / (1e-5 + pow(viewZ / 200.0, 4.0)), 1e-2, 3e3);
  var out : OITOut;
  out.accum = vec4<f32>(color * a, a) * weight;
  out.reveal = a;
  return out;
}
`;

export const PBR_SHADER = HEADER + VERTEX_STATIC + FRAGMENT;
export const PBR_SKINNED_SHADER = HEADER + VERTEX_SKINNED + FRAGMENT;
export const PBR_INSTANCED_SHADER = HEADER + VERTEX_INSTANCED + FRAGMENT;
export const PBR_MORPH_SHADER = HEADER + VERTEX_MORPH + FRAGMENT;
