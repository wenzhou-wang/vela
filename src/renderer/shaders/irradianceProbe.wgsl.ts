/** Projects an equirectangular radiance capture into nine cosine-convolved RGB SH-L2 coefficients. */
export const IRRADIANCE_PROBE_SHADER = /* wgsl */ `
const PI = 3.141592653589793;
@group(0) @binding(0) var environment : texture_2d<f32>;
@group(0) @binding(1) var environmentSampler : sampler;
@group(0) @binding(2) var<storage, read_write> coefficients : array<vec4<f32>>;

fn basis(i : u32, d : vec3<f32>) -> f32 {
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

@compute @workgroup_size(9)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= 9u) { return; }
  var sum = vec3<f32>(0.0);
  let width = 64u;
  let height = 32u;
  for (var y = 0u; y < height; y = y + 1u) {
    let v = (f32(y) + 0.5) / f32(height);
    let theta = v * PI;
    let sinTheta = sin(theta);
    for (var x = 0u; x < width; x = x + 1u) {
      let u = (f32(x) + 0.5) / f32(width);
      let phi = (u - 0.5) * 2.0 * PI;
      let d = vec3<f32>(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
      let radiance = textureSampleLevel(environment, environmentSampler, vec2<f32>(u, v), 0.0).rgb;
      let domega = (2.0 * PI / f32(width)) * (PI / f32(height)) * sinTheta;
      sum = sum + radiance * basis(i, d) * domega;
    }
  }
  let convolution = select(select(PI * 0.25, 2.0 * PI / 3.0, i <= 3u), PI, i == 0u);
  coefficients[i] = vec4<f32>(sum * convolution, 0.0);
}
`;
