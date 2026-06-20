export const DECAL_SHADER = /* wgsl */ `
struct Params { invViewProj:mat4x4<f32>, worldToDecal:mat4x4<f32>, size:vec4<f32>, color:vec4<f32> };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var scene:texture_2d<f32>;
@group(0) @binding(2) var depthTex:texture_depth_2d;
@group(0) @binding(3) var decalTex:texture_2d<f32>;
@group(0) @binding(4) var smp:sampler;
struct Out{@builtin(position) clip:vec4<f32>,@location(0) uv:vec2<f32>};
@vertex fn vs_main(@builtin(vertex_index)i:u32)->Out{var o:Out;let x=f32((i<<1u)&2u);let y=f32(i&2u);o.uv=vec2(x,y);o.clip=vec4(x*2.0-1.0,1.0-y*2.0,0.0,1.0);return o;}
@fragment fn fs_main(in:Out)->@location(0) vec4<f32>{
  let base=textureSampleLevel(scene,smp,in.uv,0.0);let d=textureLoad(depthTex,vec2<i32>(in.clip.xy),0);
  if(d>=1.0){return base;} let h=p.invViewProj*vec4<f32>(in.uv.x*2.0-1.0,1.0-in.uv.y*2.0,d,1.0);
  let local=(p.worldToDecal*vec4<f32>(h.xyz/h.w,1.0)).xyz;let halfSize=p.size.xyz*0.5;
  if(any(abs(local)>halfSize)){return base;} let uv=local.xy/p.size.xy+vec2<f32>(0.5);
  let decal=textureSampleLevel(decalTex,smp,uv,0.0)*p.color;let a=clamp(decal.a,0.0,1.0);
  return vec4<f32>(mix(base.rgb,decal.rgb,a),base.a);
}`;
