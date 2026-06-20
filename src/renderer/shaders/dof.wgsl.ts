/** Depth-aware 16-sample golden-angle bokeh resolve. */
export const DOF_SHADER = /* wgsl */ `
struct Params { resolution : vec2<f32>, nearFar : vec2<f32>, focusAperture : vec2<f32>, maxBlur : f32, _pad : f32 };
@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var scene : texture_2d<f32>;
@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var smp : sampler;
struct Out { @builtin(position) clip:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i:u32)->Out { var o:Out; let x=f32((i<<1u)&2u); let y=f32(i&2u); o.uv=vec2(x,y); o.clip=vec4(x*2.0-1.0,1.0-y*2.0,0.0,1.0); return o; }
fn linearDepth(d:f32)->f32 { return p.nearFar.x*p.nearFar.y/max(p.nearFar.y-d*(p.nearFar.y-p.nearFar.x),1e-5); }
fn coc(depth:f32)->f32 { return clamp(abs((p.focusAperture.x-depth)/max(depth,1e-4))*p.focusAperture.y*p.maxBlur,0.0,p.maxBlur); }
@fragment fn fs_main(in:Out)->@location(0) vec4<f32> {
  let centerDepth=linearDepth(textureLoad(depthTex,vec2<i32>(in.clip.xy),0));
  let radius=coc(centerDepth);
  if (radius<0.5) { return textureSampleLevel(scene,smp,in.uv,0.0); }
  var sum=vec3<f32>(0.0); var weight=0.0;
  for (var i=0;i<16;i=i+1) {
    let fi=f32(i)+0.5; let r=sqrt(fi/16.0)*radius; let a=fi*2.399963;
    let uv=in.uv+vec2(cos(a),sin(a))*r*p.resolution;
    let dims=vec2<i32>(textureDimensions(depthTex));
    let sampleDepth=linearDepth(textureLoad(depthTex,clamp(vec2<i32>(uv*vec2<f32>(dims)),vec2<i32>(0),dims-vec2<i32>(1)),0));
    let w=select(0.25,1.0,sampleDepth>=centerDepth-radius);
    sum=sum+textureSampleLevel(scene,smp,uv,0.0).rgb*w; weight=weight+w;
  }
  return vec4(sum/weight,1.0);
}`;
