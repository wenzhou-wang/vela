/**
 * Final color-grading pass: sample the tonemapped LDR color, look it up in a 3-D
 * color LUT (hardware-trilinear), and blend by strength. Runs after tonemap (and
 * FXAA) in display space — its own bind layout (the post module's binding 2 is a
 * 2-D texture, so the 3-D LUT needs a separate module).
 *
 *   0 src (tonemapped LDR) · 1 sampler · 2 lut (texture_3d) · 3 params
 *   params.info = (strength, lutSize, 0, 0); params.dmin/dmax = input domain (rgb)
 */
export const LUT_SHADER = /* wgsl */ `
@group(0) @binding(0) var src  : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var lut  : texture_3d<f32>;
struct Params {
  info : vec4<f32>,  // (strength, lutSize, 0, 0)
  dmin : vec4<f32>,  // input domain min (rgb)
  dmax : vec4<f32>,  // input domain max (rgb)
};
@group(0) @binding(3) var<uniform> params : Params;

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

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let c = clamp(textureSample(src, samp, in.uv).rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let n = params.info.y;
  // Remap the input through the LUT's domain, then to texel centers so the
  // trilinear fetch hits the table samples exactly.
  let t = clamp((c - params.dmin.rgb) / (params.dmax.rgb - params.dmin.rgb), vec3<f32>(0.0), vec3<f32>(1.0));
  let uvw = t * ((n - 1.0) / n) + (0.5 / n);
  let graded = textureSample(lut, samp, uvw).rgb;
  return vec4<f32>(mix(c, graded, params.info.x), 1.0);
}
`;
