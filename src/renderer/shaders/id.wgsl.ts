/**
 * Id-buffer shader: renders each mesh with a flat RGB color encoding a uint32
 * mesh id (r = id & 0xff, g = (id >> 8) & 0xff, b = (id >> 16) & 0xff).
 * Reads view/proj from the shared frame uniform (group 0 binding 0) and the
 * model matrix from the model uniform (group 1 binding 0).  The id value is
 * a tiny per-draw uniform in group 2 binding 0.
 */

export const ID_SHADER = /* wgsl */ `
struct Frame {
  view       : mat4x4<f32>,
  proj       : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> frame : Frame;

struct Model {
  model     : mat4x4<f32>,
  normalMat : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

@group(2) @binding(0) var<uniform> meshId : u32;

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
  let worldPos = model.model * vec4<f32>(position, 1.0);
  return frame.proj * frame.view * worldPos;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  let r = f32(meshId & 0xffu)        / 255.0;
  let g = f32((meshId >> 8u) & 0xffu)  / 255.0;
  let b = f32((meshId >> 16u) & 0xffu) / 255.0;
  return vec4<f32>(r, g, b, 1.0);
}
`;
