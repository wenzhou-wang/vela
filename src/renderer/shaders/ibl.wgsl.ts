/**
 * IBL prefilter compute shaders.
 *
 * Three entry points share a utility library:
 *   cs_brdf       — precompute the split-sum BRDF look-up table (NoV, roughness → F_scale, F_bias).
 *   cs_irradiance — convolve the environment into a cosine-weighted irradiance map.
 *   cs_specular   — GGX-importance-sample the environment for one roughness level.
 *
 * cs_brdf uses its own bind group layout (one storage texture).
 * cs_irradiance / cs_specular share a layout (env texture, sampler, output storage, params uniform).
 */
export const IBL_SHADER = /* wgsl */ `
const PI = 3.14159265358979;

// ── Utility ─────────────────────────────────────────────────────────────────

fn radicalInverse(bits : u32) -> f32 {
  var b = (bits << 16u) | (bits >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return f32(b) * 2.3283064365386963e-10;
}

fn hammersley(i : u32, N : u32) -> vec2<f32> {
  return vec2<f32>(f32(i) / f32(N), radicalInverse(i));
}

fn tbnFrame(N : vec3<f32>) -> mat3x3<f32> {
  let up  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let T   = normalize(cross(up, N));
  let B   = cross(N, T);
  return mat3x3<f32>(T, B, N);
}

fn importanceSampleGGX(xi : vec2<f32>, N : vec3<f32>, roughness : f32) -> vec3<f32> {
  let a  = roughness * roughness;
  let phi      = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  let H = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return normalize(tbnFrame(N) * H);
}

fn cosineHemisphere(xi : vec2<f32>, N : vec3<f32>) -> vec3<f32> {
  let phi      = 2.0 * PI * xi.x;
  let cosTheta = sqrt(xi.y);
  let sinTheta = sqrt(1.0 - xi.y);
  let H = vec3<f32>(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return normalize(tbnFrame(N) * H);
}

fn geometrySchlick(NoX : f32, roughness : f32) -> f32 {
  let k = roughness * roughness * 0.5;
  return NoX / (NoX * (1.0 - k) + k);
}
fn geometrySmith(NoV : f32, NoL : f32, roughness : f32) -> f32 {
  return geometrySchlick(NoV, roughness) * geometrySchlick(NoL, roughness);
}

fn dirToUv(d : vec3<f32>) -> vec2<f32> {
  return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5,
                   acos(clamp(d.y, -1.0, 1.0)) / PI);
}

// ── cs_brdf: split-sum BRDF LUT ─────────────────────────────────────────────
// group 0 binding 0 : output (NoV=u, roughness=v) → (F_scale, F_bias)

@group(0) @binding(0) var brdfOut : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_brdf(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(brdfOut);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let NoV       = (f32(gid.x) + 0.5) / f32(dims.x);
  let roughness = (f32(gid.y) + 0.5) / f32(dims.y);
  let V = vec3<f32>(sqrt(max(1.0 - NoV * NoV, 0.0)), 0.0, NoV);
  let N = vec3<f32>(0.0, 0.0, 1.0);

  const SAMPLES = 512u;
  var A = 0.0;
  var B = 0.0;
  for (var i = 0u; i < SAMPLES; i++) {
    let xi = hammersley(i, SAMPLES);
    let H  = importanceSampleGGX(xi, N, roughness);
    let L  = reflect(-V, H);
    let NoL = max(L.z, 0.0);
    let NoH = max(H.z, 0.0);
    let VoH = max(dot(V, H), 0.0);
    if (NoL > 0.0) {
      let G    = geometrySmith(NoV, NoL, roughness);
      let Gvis = G * VoH / max(NoH * NoV, 1e-4);
      let Fc   = pow(1.0 - VoH, 5.0);
      A += (1.0 - Fc) * Gvis;
      B += Fc * Gvis;
    }
  }
  textureStore(brdfOut, vec2<i32>(gid.xy),
    vec4<f32>(A / f32(SAMPLES), B / f32(SAMPLES), 0.0, 1.0));
}

// ── cs_irradiance / cs_specular shared bindings ──────────────────────────────
// group 0 binding 0 : input env (texture_2d<f32>)
// group 0 binding 1 : sampler (filtering)
// group 0 binding 2 : output storage texture
// group 0 binding 3 : IblParams (roughness, pad, outW, outH)

struct IblParams { roughness : f32, pad : f32, outW : f32, outH : f32 };

@group(0) @binding(0) var envIn    : texture_2d<f32>;
@group(0) @binding(1) var envSamp  : sampler;
@group(0) @binding(2) var iblOut   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> p : IblParams;

// ── cs_irradiance ────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn cs_irradiance(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (f32(gid.x) >= p.outW || f32(gid.y) >= p.outH) { return; }

  let uv  = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(p.outW, p.outH);
  let phi = uv.x * 2.0 * PI;
  let th  = uv.y * PI;
  let N   = vec3<f32>(sin(th) * cos(phi), cos(th), sin(th) * sin(phi));

  const SAMPLES = 64u;
  var irr = vec3<f32>(0.0);
  for (var i = 0u; i < SAMPLES; i++) {
    let xi = hammersley(i, SAMPLES);
    let L  = cosineHemisphere(xi, N);
    irr += textureSampleLevel(envIn, envSamp, dirToUv(L), 0.0).rgb;
  }
  textureStore(iblOut, vec2<i32>(gid.xy), vec4<f32>(irr / f32(SAMPLES), 1.0));
}

// ── cs_specular ──────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn cs_specular(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (f32(gid.x) >= p.outW || f32(gid.y) >= p.outH) { return; }

  let uv  = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(p.outW, p.outH);
  let phi = uv.x * 2.0 * PI;
  let th  = uv.y * PI;
  let N   = vec3<f32>(sin(th) * cos(phi), cos(th), sin(th) * sin(phi));
  let roughness = p.roughness;

  // For roughness=0 just mirror-reflect the env.
  if (roughness < 0.001) {
    let c = textureSampleLevel(envIn, envSamp, dirToUv(N), 0.0).rgb;
    textureStore(iblOut, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
    return;
  }

  const SAMPLES = 128u;
  var result = vec3<f32>(0.0);
  var totalW  = 0.0;
  for (var i = 0u; i < SAMPLES; i++) {
    let xi = hammersley(i, SAMPLES);
    let H  = importanceSampleGGX(xi, N, roughness);
    let L  = reflect(-N, H);  // V = N (split-sum assumption)
    let NoL = max(dot(N, L), 0.0);
    if (NoL > 0.0) {
      result += textureSampleLevel(envIn, envSamp, dirToUv(L), 0.0).rgb * NoL;
      totalW += NoL;
    }
  }
  textureStore(iblOut, vec2<i32>(gid.xy),
    vec4<f32>(result / max(totalW, 1e-4), 1.0));
}
`;
