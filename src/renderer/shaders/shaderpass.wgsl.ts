/**
 * Wrapper WGSL for a user `ShaderPass`. Group 0 carries the previous-stage
 * color, the scene depth, and built-in params; group 1 holds the user's
 * auto-packed uniforms (`computeUniformLayout(..., 1)`). The user defines
 * `fn effect(uv : vec2<f32>) -> vec4<f32>`; this module supplies the
 * fullscreen-triangle vertex stage and the `sceneColor`/`sceneDepth` helpers.
 */

const PASS_HEADER = /* wgsl */ `
struct PassParams {
  resolution : vec4<f32>,  // xy = pixels, zw = 1/pixels
  time       : vec4<f32>,  // x = elapsed seconds
};

@group(0) @binding(0) var sceneTex : texture_2d<f32>;
@group(0) @binding(1) var sceneSmp : sampler;
@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var<uniform> pp : PassParams;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
  var out : VSOut;
  let x = f32((vi << 1u) & 2u);
  let y = f32(vi & 2u);
  out.uv = vec2(x, y);
  out.clip = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

fn sceneColor(uv : vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(sceneTex, sceneSmp, uv, 0.0);
}

fn sceneDepth(uv : vec2<f32>) -> f32 {
  let dim = vec2<i32>(pp.resolution.xy);
  let px = clamp(vec2<i32>(uv * pp.resolution.xy), vec2<i32>(0), dim - vec2<i32>(1));
  return textureLoad(depthTex, px, 0);
}
`;

const PASS_MAIN = /* wgsl */ `
@fragment fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  return effect(in.uv);
}
`;

/** Assemble a complete ShaderPass module: header + user group-1 uniforms + effect. */
export function buildShaderPass(uniformWgsl: string, effectCode: string): string {
  return (
    PASS_HEADER +
    uniformWgsl +
    '\n// --- user effect code ---\n' +
    effectCode +
    '\n// --- end user effect code ---\n' +
    PASS_MAIN
  );
}
