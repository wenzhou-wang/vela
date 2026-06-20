/** 64-bin log-luminance histogram and temporal exposure reduction. */
export const EXPOSURE_SHADER = /* wgsl */ `
@group(0) @binding(0) var hdr : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram : array<atomic<u32>, 64>;
struct Params { sizeDtDet : vec4<f32>, evSpeed : vec4<f32> };
@group(0) @binding(2) var<uniform> p : Params;
@group(0) @binding(3) var<storage, read_write> exposure : array<f32>;

@compute @workgroup_size(8, 8)
fn cs_histogram(@builtin(global_invocation_id) gid : vec3<u32>) {
  let size=vec2<u32>(p.sizeDtDet.xy); if (any(gid.xy>=size)) { return; }
  let c=textureLoad(hdr,vec2<i32>(gid.xy),0).rgb;
  let lum=max(dot(c,vec3<f32>(0.2126,0.7152,0.0722)),1e-6);
  let t=clamp((log2(lum)-p.evSpeed.x)/(p.evSpeed.y-p.evSpeed.x),0.0,0.9999);
  atomicAdd(&histogram[u32(t*64.0)],1u);
}

@compute @workgroup_size(1)
fn cs_reduce() {
  var weighted=0.0; var count=0u;
  for(var i=0u;i<64u;i=i+1u){let n=atomicLoad(&histogram[i]); weighted+=f32(n)*(f32(i)+0.5); count+=n;}
  if(count==0u){exposure[0]=1.0;return;}
  let bin=weighted/f32(count); let ev=mix(p.evSpeed.x,p.evSpeed.y,bin/64.0);
  let target=clamp(0.18/exp2(ev),exp2(-p.evSpeed.y),exp2(-p.evSpeed.x));
  let alpha=select(1.0,1.0-exp(-p.sizeDtDet.z*p.evSpeed.z),p.sizeDtDet.w<0.5);
  exposure[0]=mix(exposure[0],target,alpha);
}`;
