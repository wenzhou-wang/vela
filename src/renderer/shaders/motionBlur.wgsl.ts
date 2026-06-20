/** Velocity-neighborhood max plus nine-tap reconstruction blur in HDR. */
export const MOTION_BLUR_SHADER = /* wgsl */ `
struct Params { resolutionStrength : vec4<f32> };
@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var scene : texture_2d<f32>;
@group(0) @binding(2) var velocity : texture_2d<f32>;
@group(0) @binding(3) var smp : sampler;
struct Out { @builtin(position) clip : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i : u32) -> Out {
  var o : Out; let x=f32((i<<1u)&2u); let y=f32(i&2u);
  o.uv=vec2<f32>(x,y); o.clip=vec4<f32>(x*2.0-1.0,1.0-y*2.0,0.0,1.0); return o;
}
@fragment fn fs_main(in : Out) -> @location(0) vec4<f32> {
  let texel = p.resolutionStrength.xy;
  var v = vec2<f32>(0.0);
  for (var y=-1; y<=1; y=y+1) { for (var x=-1; x<=1; x=x+1) {
    let candidate=textureSampleLevel(velocity,smp,in.uv+vec2<f32>(f32(x),f32(y))*texel,0.0).xy;
    if (dot(candidate,candidate)>dot(v,v)) { v=candidate; }
  }}
  v = v * p.resolutionStrength.z;
  var color=vec3<f32>(0.0); var weight=0.0;
  for (var i=0; i<9; i=i+1) {
    let t=f32(i)/8.0-0.5; let w=1.0-abs(t)*1.5;
    color=color+textureSampleLevel(scene,smp,in.uv+v*t,0.0).rgb*w; weight=weight+w;
  }
  return vec4<f32>(color/weight,1.0);
}`;
