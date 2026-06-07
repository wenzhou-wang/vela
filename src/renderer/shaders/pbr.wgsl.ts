/**
 * Physically based metallic-roughness shader (glTF 2.0 model).
 * - Cook-Torrance specular: GGX NDF, Smith height-correlated visibility, Schlick Fresnel
 * - Lambertian diffuse
 * - Tangent-space normal mapping (safe with flat default normal map)
 * - ACES-approx filmic tonemap + linear->sRGB on output
 *
 * Single "uber" shader: missing material maps are bound as white/flat defaults,
 * so no per-texture pipeline variants are needed.
 */
export const PBR_SHADER = /* wgsl */ `
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
  // xyz = world position (point) or direction-target base, w = kind
  positionKind : vec4<f32>,
  // xyz = normalized direction (directional), w = range (point, 0 = infinite)
  directionRange : vec4<f32>,
  // xyz = color * intensity, w = decay
  colorDecay : vec4<f32>,
};

struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
};

struct MaterialU {
  baseColor : vec4<f32>,        // rgb + opacity
  emissive : vec4<f32>,         // rgb + emissiveIntensity
  params : vec4<f32>,           // metalness, roughness, normalScale, occlusionStrength
  misc : vec4<f32>,             // alphaCutoff, flags, pad, pad
};

@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var<storage, read> lights : array<Light>;

@group(1) @binding(0) var<uniform> model : Model;

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

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent : vec4<f32>,
};

struct VSOut {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) worldPos : vec3<f32>,
  @location(1) worldNormal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) worldTangent : vec3<f32>,
  @location(4) tangentSign : f32,
};

@vertex
fn vs_main(in : VSIn) -> VSOut {
  var out : VSOut;
  let worldPos4 = model.model * vec4<f32>(in.position, 1.0);
  out.worldPos = worldPos4.xyz;
  out.clipPosition = frame.proj * frame.view * worldPos4;

  let n = normalize((model.normalMat * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldNormal = n;
  out.worldTangent = normalize((model.model * vec4<f32>(in.tangent.xyz, 0.0)).xyz);
  out.tangentSign = in.tangent.w;
  out.uv = in.uv;
  return out;
}

// ---- BRDF helpers ----

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

  // Base color (sRGB texture decoded by hardware sampler)
  let baseSample = textureSample(baseColorTex, baseColorSmp, in.uv);
  let baseColor = material.baseColor.rgb * baseSample.rgb;
  let alpha = material.baseColor.a * baseSample.a;

  let alphaCutoff = material.misc.x;
  if (alphaCutoff > 0.0 && alpha < alphaCutoff) {
    discard;
  }

  // Metalness / roughness (linear). glTF: G = roughness, B = metalness.
  let mrSample = textureSample(mrTex, mrSmp, in.uv);
  let metalness = clamp(material.params.x * mrSample.b, 0.0, 1.0);
  var roughness = clamp(material.params.y * mrSample.g, 0.04, 1.0);

  // Geometric + mapped normal (flip for back faces of double-sided materials)
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

  // Dielectric F0 = 0.04, metals tint specular with base color
  let f0 = mix(vec3<f32>(0.04), baseColor, metalness);
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

    radiance = radiance * attenuation * NoL;
    color = color + (diffuse + specular) * radiance;
  }

  // Ambient (flat irradiance) modulated by occlusion. AO affects only the
  // indirect/ambient term, not direct lighting.
  let aoSample = textureSample(occlusionTex, occlusionSmp, in.uv).r;
  let ao = mix(1.0, aoSample, material.params.w);
  color = color + frame.ambient.rgb * diffuseColor * ao;

  // Emissive
  let emissiveSample = textureSample(emissiveTex, emissiveSmp, in.uv).rgb;
  color = color + material.emissive.rgb * material.emissive.a * emissiveSample;

  // Exposure, tonemap, sRGB encode
  color = color * frame.ambient.w;
  color = acesFilmic(color);
  color = linearToSRGB(color);

  return vec4<f32>(color, alpha);
}
`;
