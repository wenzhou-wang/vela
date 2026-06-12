import { HEADER, SHADE_HELPERS, VERTEX_STATIC, VERTEX_INSTANCED, VERTEX_SKINNED, VERTEX_MORPH } from './pbr.wgsl';
import type { PipelineVariant } from '../PipelineCache';

/**
 * WGSL assembly for `ShaderMaterial`: the user supplies a `surface()` function
 * that fills a `Surface` struct; the engine wraps it with the shared vertex
 * stages and the full PBR lighting/shadow/IBL/cluster pipeline from pbr.wgsl.
 *
 * The generated module replaces the StandardMaterial bind group (group 2) with
 * a single auto-packed uniform buffer exposed to user code as `u`.
 */

// Everything user code can rely on: the Surface output struct, engine-filled
// defaults, and an elapsed-time helper.
const SURFACE_PRELUDE = /* wgsl */ `
// The shading inputs your surface() function produces. Fields you don't set
// keep the values from defaultSurface(in).
struct Surface {
  baseColor : vec3<f32>,  // albedo (linear)
  alpha     : f32,        // opacity (used when material.transparent)
  metalness : f32,        // 0 = dielectric, 1 = metal
  roughness : f32,        // 0.04..1 perceptual roughness
  emissive  : vec3<f32>,  // emitted light (linear, HDR ok)
  normal    : vec3<f32>,  // world-space shading normal
  occlusion : f32,        // ambient occlusion factor 0..1
};

// Sensible PBR defaults; start from this and override what you need.
fn defaultSurface(in : VSOut) -> Surface {
  var s : Surface;
  s.baseColor = vec3<f32>(0.8);
  s.alpha = 1.0;
  s.metalness = 0.0;
  s.roughness = 0.5;
  s.emissive = vec3<f32>(0.0);
  s.normal = in.worldNormal;
  s.occlusion = 1.0;
  return s;
}

// Seconds since the renderer started (uploaded every frame).
fn elapsedTime() -> f32 {
  return frame.clusterDims.z;
}
`;

// Lit shading + the two fragment entry points, mirroring the PBR fragment's
// exposure/tonemap and OIT weighting so ShaderMaterial meshes composite
// identically to StandardMaterial ones in every render path.
const SURFACE_SHADE = /* wgsl */ `
fn smShade(in : VSOut, frontFacing : bool) -> vec4<f32> {
  var s = surface(in);
  var N = normalize(s.normal);
  if (!frontFacing) { N = -N; }

  let V = normalize(frame.cameraPos.xyz - in.worldPos);
  let NoV = max(dot(N, V), 1e-4);
  let roughness = clamp(s.roughness, 0.04, 1.0);
  let metalness = clamp(s.metalness, 0.0, 1.0);
  let f0 = mix(vec3<f32>(0.04), s.baseColor, metalness);
  let diffuseColor = s.baseColor * (1.0 - metalness);

  var color = vec3<f32>(0.0);
  let list = lightList(in.clipPosition.xy, in.worldPos);
  for (var j = 0u; j < list.x; j = j + 1u) {
    let i = lightIndex(list, j);
    let lc = evaluateLight(i, in.worldPos, N);
    let L = lc.L;
    let H = normalize(V + L);
    let NoL = max(dot(N, L), 0.0);
    if (NoL <= 0.0) { continue; }
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    let D = distributionGGX(NoH, roughness);
    let Vis = visibilitySmith(NoV, NoL, roughness);
    let F = fresnelSchlick(VoH, f0);
    let lit = (vec3<f32>(1.0) - F) * diffuseColor / PI + D * Vis * F;
    color = color + lit * lc.radiance * NoL;
  }

  color = color + indirectLight(N, V, NoV, roughness, f0, diffuseColor, clamp(s.occlusion, 0.0, 1.0));
  color = color + s.emissive;
  color = applyFog(color, in.worldPos);
  return vec4<f32>(color, clamp(s.alpha, 0.0, 1.0));
}

@fragment
fn fs_main(in : VSOut, @builtin(front_facing) frontFacing : bool) -> @location(0) vec4<f32> {
  let s = smShade(in, frontFacing);
  var color = s.rgb * frame.ambient.w; // exposure
  // envParams.w bit 0: linear output (post pipeline tonemaps later).
  if ((u32(frame.envParams.w) & 1u) == 0u) {
    color = acesFilmic(color);
    color = linearToSRGB(color);
  }
  return vec4<f32>(color, s.a);
}

struct OITOut {
  @location(0) accum : vec4<f32>,
  @location(1) reveal : f32,
};

@fragment
fn fs_oit(in : VSOut, @builtin(front_facing) frontFacing : bool) -> OITOut {
  let s = smShade(in, frontFacing);
  let color = s.rgb * frame.ambient.w;
  let a = clamp(s.a, 0.0, 1.0);
  let viewZ = abs((frame.view * vec4<f32>(in.worldPos, 1.0)).z);
  let weight = a * clamp(0.03 / (1e-5 + pow(viewZ / 200.0, 4.0)), 1e-2, 3e3);
  var out : OITOut;
  out.accum = vec4<f32>(color * a, a) * weight;
  out.reveal = a;
  return out;
}
`;

const VERTEX_BY_VARIANT: Record<PipelineVariant, string> = {
  static: VERTEX_STATIC,
  skinned: VERTEX_SKINNED,
  instanced: VERTEX_INSTANCED,
  morph: VERTEX_MORPH,
};

/**
 * Assemble the complete WGSL module for a ShaderMaterial.
 * `uniformStruct` is the generated `struct SMUniforms {...}` + binding (may be
 * empty when the material declares no uniforms); `surfaceCode` must define
 * `fn surface(in : VSOut) -> Surface`.
 */
export function buildSurfaceShader(
  variant: PipelineVariant,
  uniformStruct: string,
  surfaceCode: string,
): string {
  return (
    HEADER +
    VERTEX_BY_VARIANT[variant] +
    SHADE_HELPERS +
    SURFACE_PRELUDE +
    uniformStruct +
    '\n// --- user surface code ---\n' +
    surfaceCode +
    '\n// --- end user surface code ---\n' +
    SURFACE_SHADE
  );
}
