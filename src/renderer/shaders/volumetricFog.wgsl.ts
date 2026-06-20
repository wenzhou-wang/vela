import { FRAME_DEFS } from './pbr.wgsl';

/** Low-resolution clustered froxel integration for volumetric fog. */
export const VOLUMETRIC_FOG_SHADER = FRAME_DEFS + /* wgsl */ `
@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var<storage, read> lights : array<Light>;
@group(0) @binding(2) var<storage, read> clusterLights : array<u32>;
struct Cascades { viewProj : array<mat4x4<f32>, 4>, splits : vec4<f32>, params : vec4<f32> };
@group(0) @binding(3) var<uniform> cascades : Cascades;
@group(0) @binding(4) var shadowMap : texture_depth_2d;
@group(0) @binding(5) var shadowSampler : sampler_comparison;
@group(0) @binding(6) var volume : texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var<uniform> invViewProj : mat4x4<f32>;

fn shadowVisibility(worldPos : vec3<f32>, viewDepth : f32) -> f32 {
  if (frame.shadowParams.x < 0.5) { return 1.0; }
  let count = u32(cascades.params.x);
  var c = 0u;
  while (c + 1u < count && viewDepth > cascades.splits[c]) { c = c + 1u; }
  let lp = cascades.viewProj[c] * vec4<f32>(worldPos, 1.0);
  let ndc = lp.xyz / lp.w;
  if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) { return 1.0; }
  let scale = select(1.0, 0.5, count > 1u);
  let offset = vec2<f32>(f32(c % 2u), f32(c / 2u)) * scale;
  let uv = offset + (ndc.xy * vec2<f32>(0.5, -0.5) + 0.5) * scale;
  return textureSampleCompareLevel(shadowMap, shadowSampler, uv, ndc.z);
}

fn lightAt(i : u32, p : vec3<f32>, viewDepth : f32) -> vec3<f32> {
  let light = lights[i];
  let kind = u32(light.positionKind.w);
  var attenuation = 1.0;
  if (kind == LIGHT_DIRECTIONAL) {
    if (i == u32(frame.shadowParams.w)) { attenuation = shadowVisibility(p, viewDepth); }
  } else {
    let delta = light.positionKind.xyz - p;
    let dist = length(delta);
    attenuation = 1.0 / max(pow(dist, light.colorDecay.w), 1e-4);
    if (light.directionRange.w > 0.0) {
      attenuation = attenuation * pow(clamp(1.0 - pow(dist / light.directionRange.w, 4.0), 0.0, 1.0), 2.0);
    }
    if (kind == LIGHT_SPOT) {
      let L = delta / max(dist, 1e-4);
      let cosine = dot(-L, normalize(light.directionRange.xyz));
      let cone = clamp((cosine - light.spotParams.y) / max(light.spotParams.x - light.spotParams.y, 1e-4), 0.0, 1.0);
      attenuation = attenuation * cone * cone;
    }
  }
  return light.colorDecay.xyz * attenuation;
}

@compute @workgroup_size(4, 3, 4)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(volume);
  if (any(gid >= dims)) { return; }
  let near = frame.clusterParams.y;
  let far = frame.clusterParams.z;
  let z01 = (f32(gid.z) + 0.5) / f32(dims.z);
  let viewDepth = mix(near, far, z01);
  let ndc = vec2<f32>((f32(gid.x) + 0.5) / f32(dims.x) * 2.0 - 1.0,
                      1.0 - (f32(gid.y) + 0.5) / f32(dims.y) * 2.0);
  let farH = invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  let farWorld = farH.xyz / farH.w;
  let ray = normalize(farWorld - frame.cameraPos.xyz);
  let viewRay = frame.view * vec4<f32>(ray, 0.0);
  let rayLength = viewDepth / max(-viewRay.z, 1e-4);
  let stepLength = rayLength / 8.0;
  let clusterZ = min(u32(log(max(viewDepth, near) / near) / log(far / near) * f32(dims.z)), dims.z - 1u);
  let clusterBase = (gid.x + gid.y * dims.x + clusterZ * dims.x * dims.y) * (MAX_PER_CLUSTER + 1u);
  let clustered = frame.clusterParams.x > 0.5;
  let count = select(u32(frame.cameraPos.w), min(clusterLights[clusterBase], MAX_PER_CLUSTER), clustered);
  var transmittance = 1.0;
  var scattering = vec3<f32>(0.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    let t = (f32(step) + 0.5) * stepLength;
    let p = frame.cameraPos.xyz + ray * t;
    let sampleDepth = viewDepth * (f32(step) + 0.5) / 8.0;
    var extinction = 2.0 * frame.fogParams.x * frame.fogParams.x * sampleDepth;
    if (frame.fogColor.w < 1.5) {
      extinction = select(0.0, 3.0 / max(frame.fogParams.y - frame.fogParams.x, 1e-4), sampleDepth >= frame.fogParams.x);
    }
    if (frame.fogParams.z > 0.0) { extinction = extinction * exp(-frame.fogParams.z * max(p.y, 0.0)); }
    var light = frame.ambient.rgb;
    for (var j = 0u; j < count; j = j + 1u) {
      let index = select(j, clusterLights[clusterBase + 1u + j], clustered);
      light = light + lightAt(index, p, sampleDepth) * 0.08;
    }
    let opacity = 1.0 - exp(-extinction * stepLength);
    scattering = scattering + transmittance * opacity * (frame.fogColor.rgb + light);
    transmittance = transmittance * (1.0 - opacity);
  }
  textureStore(volume, gid, vec4<f32>(scattering, transmittance));
}
`;
