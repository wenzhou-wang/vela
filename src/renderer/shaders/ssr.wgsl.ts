/**
 * Screen-space reflections in HDR space.
 *
 * Rays start at the packed scene normal/depth surface, march in view space,
 * and test against the current frame's max-depth hi-Z pyramid. A hit samples
 * the HDR scene; a miss returns the original color, whose PBR specular already
 * contains the environment fallback.
 */
export const SSR_SHADER = /* wgsl */ `
struct SSRParams {
  proj       : mat4x4<f32>,
  invProj    : mat4x4<f32>,
  view       : mat4x4<f32>,
  resolution : vec4<f32>, // xy = pixels, zw = inverse pixels
  ray        : vec4<f32>, // x = max distance, y = thickness, z = intensity, w = hi-Z mip count
};

@group(0) @binding(0) var<uniform> p : SSRParams;
@group(0) @binding(1) var sceneTex : texture_2d<f32>;
@group(0) @binding(2) var normalDepthTex : texture_2d<f32>;
@group(0) @binding(3) var hiZTex : texture_2d<f32>;
@group(0) @binding(4) var linearSmp : sampler;
@group(0) @binding(5) var nearestSmp : sampler;

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

fn loadDepth(uv : vec2<f32>, mip : f32) -> f32 {
  return textureSampleLevel(hiZTex, nearestSmp, uv, mip).r;
}

fn viewPosition(uv : vec2<f32>, depth : f32) -> vec3<f32> {
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
  let h = p.invProj * ndc;
  return h.xyz / h.w;
}

fn projectView(position : vec3<f32>) -> vec3<f32> {
  let clip = p.proj * vec4<f32>(position, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn safeNormal(v : vec3<f32>) -> vec3<f32> {
  return v * inverseSqrt(max(dot(v, v), 1e-8));
}

@fragment fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let base = textureSampleLevel(sceneTex, linearSmp, in.uv, 0.0);
  let depth = loadDepth(in.uv, 0.0);
  if (depth >= 1.0) { return base; }

  let packed = textureLoad(
    normalDepthTex,
    clamp(vec2<i32>(in.uv * p.resolution.xy), vec2<i32>(0), vec2<i32>(p.resolution.xy) - vec2<i32>(1)),
    0,
  );
  let worldNormal = safeNormal(packed.xyz);
  if (dot(worldNormal, worldNormal) < 0.5) { return base; }

  let viewPos = viewPosition(in.uv, depth);
  let viewNormal = safeNormal((p.view * vec4<f32>(worldNormal, 0.0)).xyz);
  var incident = safeNormal(viewPos);
  if (p.proj[3][3] > 0.5) { incident = vec3<f32>(0.0, 0.0, -1.0); }
  let rayDir = safeNormal(reflect(incident, viewNormal));
  // A reflected ray heading behind the camera cannot hit the visible buffer.
  if (rayDir.z >= -1e-4) { return base; }

  let maxDistance = max(p.ray.x, 0.01);
  let thickness = max(p.ray.y, 1e-4);
  let origin = viewPos + viewNormal * thickness;
  var previousT = 0.0;
  var previousUV = in.uv;
  var hitUV = vec2<f32>(0.0);
  var hitT = 0.0;
  var hit = false;

  // Quadratic spacing keeps detail near the surface while still tracing long rays.
  for (var i = 1u; i <= 48u; i = i + 1u) {
    let f = f32(i) / 48.0;
    let t = max(thickness, f * f * maxDistance);
    let rayPos = origin + rayDir * t;
    let projected = projectView(rayPos);
    let uv = projected.xy;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) {
      break;
    }

    let footprint = max(length((uv - previousUV) * p.resolution.xy), 1.0);
    let mip = clamp(floor(log2(footprint)), 0.0, max(p.ray.w - 1.0, 0.0));
    let sceneDepth = loadDepth(uv, mip);
    if (sceneDepth < 1.0) {
      let scenePos = viewPosition(uv, sceneDepth);
      if (-rayPos.z >= -scenePos.z - thickness) {
        // Descend to mip 0 with a short binary search between the last two samples.
        var lo = previousT;
        var hi = t;
        for (var j = 0u; j < 5u; j = j + 1u) {
          let mid = (lo + hi) * 0.5;
          let midPos = origin + rayDir * mid;
          let midProjected = projectView(midPos);
          let midDepth = loadDepth(midProjected.xy, 0.0);
          let midScene = viewPosition(midProjected.xy, midDepth);
          if (midDepth < 1.0 && -midPos.z >= -midScene.z) { hi = mid; }
          else { lo = mid; }
        }
        let candidate = origin + rayDir * hi;
        let candidateUV = projectView(candidate).xy;
        let candidateDepth = loadDepth(candidateUV, 0.0);
        if (candidateDepth < 1.0) {
          let candidateScene = viewPosition(candidateUV, candidateDepth);
          let pixelTravel = length((candidateUV - in.uv) * p.resolution.xy);
          if (pixelTravel > 2.0 && abs(candidate.z - candidateScene.z) <= thickness * 2.0) {
            hit = true;
            hitUV = candidateUV;
            hitT = hi;
            break;
          }
        }
      }
    }
    previousT = t;
    previousUV = uv;
  }

  if (!hit) { return base; }
  let reflected = textureSampleLevel(sceneTex, linearSmp, hitUV, 0.0).rgb;
  let edge = min(min(hitUV.x, 1.0 - hitUV.x), min(hitUV.y, 1.0 - hitUV.y));
  let edgeFade = smoothstep(0.0, 0.08, edge);
  let distanceFade = 1.0 - clamp(hitT / maxDistance, 0.0, 1.0);
  let NoV = clamp(dot(viewNormal, -incident), 0.0, 1.0);
  let fresnel = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);
  let weight = max(p.ray.z, 0.0) * fresnel * edgeFade * distanceFade;
  return vec4<f32>(base.rgb + reflected * weight, base.a);
}
`;
