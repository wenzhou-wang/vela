/** Converts six cube-face render targets (+X,-X,+Y,-Y,+Z,-Z) to equirectangular. */
export const REFLECTION_PROBE_SHADER = /* wgsl */ `
const PI = 3.141592653589793;

@group(0) @binding(0) var facePX : texture_2d<f32>;
@group(0) @binding(1) var faceNX : texture_2d<f32>;
@group(0) @binding(2) var facePY : texture_2d<f32>;
@group(0) @binding(3) var faceNY : texture_2d<f32>;
@group(0) @binding(4) var facePZ : texture_2d<f32>;
@group(0) @binding(5) var faceNZ : texture_2d<f32>;
@group(0) @binding(6) var faceSampler : sampler;
@group(0) @binding(7) var outputTex : texture_storage_2d<rgba16float, write>;

fn sampleCube(d : vec3<f32>) -> vec3<f32> {
  let a = abs(d);
  var uv : vec2<f32>;
  var c : vec3<f32>;
  if (a.x >= a.y && a.x >= a.z) {
    if (d.x > 0.0) { uv = vec2<f32>(-d.z, -d.y) / a.x; c = textureSampleLevel(facePX, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
    else { uv = vec2<f32>(d.z, -d.y) / a.x; c = textureSampleLevel(faceNX, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
  } else if (a.y >= a.z) {
    if (d.y > 0.0) { uv = vec2<f32>(d.x, d.z) / a.y; c = textureSampleLevel(facePY, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
    else { uv = vec2<f32>(d.x, -d.z) / a.y; c = textureSampleLevel(faceNY, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
  } else {
    if (d.z > 0.0) { uv = vec2<f32>(d.x, -d.y) / a.z; c = textureSampleLevel(facePZ, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
    else { uv = vec2<f32>(-d.x, -d.y) / a.z; c = textureSampleLevel(faceNZ, faceSampler, uv * 0.5 + 0.5, 0.0).rgb; }
  }
  return c;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let size = textureDimensions(outputTex);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
  let phi = (uv.x - 0.5) * 2.0 * PI;
  let theta = uv.y * PI;
  let d = vec3<f32>(cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta));
  textureStore(outputTex, gid.xy, vec4<f32>(sampleCube(d), 1.0));
}
`;
