/**
 * Depth-only shader for the directional shadow pass: transforms positions by the
 * light's view-projection and the per-object model matrix, writing only depth.
 * Reuses the model bind group (group 1); group 0 carries the light matrix.
 */
export const SHADOW_SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> lightViewProj : mat4x4<f32>;

struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
  return lightViewProj * model.model * vec4<f32>(position, 1.0);
}
`;

/** Position-only vertex layout for the shadow pass. */
export const SHADOW_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
];
