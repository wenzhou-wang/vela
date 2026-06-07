/**
 * Unlit line shader for debug helpers (grid / axes / box). Transforms positions
 * by the per-object model matrix and the frame view/projection, multiplies a flat
 * material color by an optional per-vertex color, and sRGB-encodes for the
 * non-srgb swap-chain target (no tonemap — helpers should read at literal color).
 *
 * Reuses bind group 0 (frame) and group 1 (model) from the PBR layout; group 2
 * is a small line-material uniform (color + flags).
 */
export const LINE_SHADER = /* wgsl */ `
struct Frame {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  cameraPos : vec4<f32>,
  ambient : vec4<f32>,
  lightViewProj : mat4x4<f32>,
  shadowParams : vec4<f32>,
  envParams : vec4<f32>, // w = linear output (post pipeline tonemaps later)
};
@group(0) @binding(0) var<uniform> frame : Frame;

struct Model {
  model : mat4x4<f32>,
  normalMat : mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> model : Model;

struct LineU {
  color : vec4<f32>,  // rgb (linear) + opacity
  flags : vec4<f32>,  // x = use vertex colors
};
@group(2) @binding(0) var<uniform> lineMat : LineU;

struct VSIn {
  @location(0) position : vec3<f32>,
  @location(1) color : vec4<f32>,
};

struct VSOut {
  @builtin(position) clipPosition : vec4<f32>,
  @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(in : VSIn) -> VSOut {
  var out : VSOut;
  let world = model.model * vec4<f32>(in.position, 1.0);
  out.clipPosition = frame.proj * frame.view * world;
  var c = lineMat.color;
  if (lineMat.flags.x > 0.5) {
    c = c * in.color;
  }
  out.color = c;
  return out;
}

fn linearToSRGB(c : vec3<f32>) -> vec3<f32> {
  let cutoff = step(vec3<f32>(0.0031308), c);
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, cutoff);
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  if (frame.envParams.w >= 0.5) {
    return vec4<f32>(in.color.rgb, in.color.a); // linear output for the post pipeline
  }
  return vec4<f32>(linearToSRGB(in.color.rgb), in.color.a);
}
`;

/** Line vertex streams: position (vec3) + color (vec4). */
export const LINE_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
  { arrayStride: 16, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }] },
];
