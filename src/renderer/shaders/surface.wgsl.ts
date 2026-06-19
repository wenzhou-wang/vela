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

// Per-light terms for an optional fn light(s : Surface, l : LightSample) ->
// vec3<f32>. The engine evaluates which lights reach the fragment (clustered
// list, type, distance/cone attenuation, shadow visibility) and calls light()
// once per light; return that light's contribution and the engine sums them.
// Remap NoL for toon ramps, wrap/half-Lambert, posterized highlights, etc.
struct LightSample {
  L : vec3<f32>,        // world-space direction from the surface toward the light
  radiance : vec3<f32>, // light color × distance/cone attenuation × shadow visibility
  N : vec3<f32>,        // shading normal (already flipped for back faces)
  V : vec3<f32>,        // direction to the camera
  H : vec3<f32>,        // half vector normalize(V + L)
  NoL : f32,            // raw dot(N, L); may be negative (for wrap/half-Lambert)
  NoV : f32,
  NoH : f32,            // max(dot(N, H), 0)
  VoH : f32,            // max(dot(V, H), 0)
};

// Indirect (ambient / image-based) inputs for an optional
// fn ambient(s : Surface, ind : IndirectSample) -> vec3<f32>.
struct IndirectSample {
  N : vec3<f32>,
  V : vec3<f32>,
  NoV : f32,
};

// The engine's default Cook-Torrance contribution for one light. Call it from a
// custom light() to keep physical lighting and add on top, or ignore it.
fn defaultLight(s : Surface, l : LightSample) -> vec3<f32> {
  let NoL = max(l.NoL, 0.0);
  if (NoL <= 0.0) { return vec3<f32>(0.0); }
  let roughness = clamp(s.roughness, 0.04, 1.0);
  let metalness = clamp(s.metalness, 0.0, 1.0);
  let f0 = mix(vec3<f32>(0.04), s.baseColor, metalness);
  let diffuseColor = s.baseColor * (1.0 - metalness);
  let D = distributionGGX(l.NoH, roughness);
  let Vis = visibilitySmith(l.NoV, NoL, roughness);
  let F = fresnelSchlick(l.VoH, f0);
  let lit = (vec3<f32>(1.0) - F) * diffuseColor / PI + D * Vis * F;
  return lit * l.radiance * NoL;
}

// The engine's default indirect light (image-based when an environment is bound,
// flat ambient otherwise). Call it from a custom ambient() to blend or replace.
fn defaultIndirect(s : Surface, ind : IndirectSample) -> vec3<f32> {
  let roughness = clamp(s.roughness, 0.04, 1.0);
  let metalness = clamp(s.metalness, 0.0, 1.0);
  let f0 = mix(vec3<f32>(0.04), s.baseColor, metalness);
  let diffuseColor = s.baseColor * (1.0 - metalness);
  return indirectLight(ind.N, ind.V, ind.NoV, roughness, f0, diffuseColor, clamp(s.occlusion, 0.0, 1.0));
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

  var color = vec3<f32>(0.0);
  let list = lightList(in.clipPosition.xy, in.worldPos);
  for (var j = 0u; j < list.x; j = j + 1u) {
    let i = lightIndex(list, j);
    let lc = evaluateLight(i, in.worldPos, N);
    var l : LightSample;
    l.L = lc.L;
    l.radiance = lc.radiance;
    l.N = N;
    l.V = V;
    l.H = normalize(V + l.L);
    l.NoL = dot(N, l.L);
    l.NoV = NoV;
    l.NoH = max(dot(N, l.H), 0.0);
    l.VoH = max(dot(V, l.H), 0.0);
    color = color + velaLight(s, l);
  }

  var ind : IndirectSample;
  ind.N = N;
  ind.V = V;
  ind.NoV = NoV;
  color = color + velaIndirect(s, ind);
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

// The expression a vertex displacement hook replaces in the vertex stages.
const POSITION_EXPR = 'vec4<f32>(in.position, 1.0)';

/**
 * Assemble the complete WGSL module for a ShaderMaterial.
 * `uniformStruct` is the generated group-2 declarations (scalar buffer +
 * texture bindings; may be empty); `surfaceCode` must define
 * `fn surface(in : VSOut) -> Surface`; `vertexCode` optionally defines
 * `fn displace(position : vec3<f32>, in : VSIn) -> vec3<f32>`, spliced into
 * the static/instanced vertex stages before the model transform.
 * `lightCode`/`ambientCode` optionally define
 * `fn light(s : Surface, l : LightSample) -> vec3<f32>` (called per reaching
 * light) and `fn ambient(s : Surface, ind : IndirectSample) -> vec3<f32>`
 * (called once for indirect light); absent, the engine uses its PBR defaults.
 */
export function buildSurfaceShader(
  variant: PipelineVariant,
  uniformStruct: string,
  surfaceCode: string,
  vertexCode?: string | null,
  lightCode?: string | null,
  ambientCode?: string | null,
): string {
  let vertex = VERTEX_BY_VARIANT[variant];
  let userVertex = '';
  if (vertexCode && (variant === 'static' || variant === 'instanced')) {
    if (!vertex.includes(POSITION_EXPR)) {
      throw new Error(
        `buildSurfaceShader: vertex stage "${variant}" no longer contains the ` +
        `position expression the displacement hook splices into — update POSITION_EXPR.`,
      );
    }
    vertex = vertex.replace(POSITION_EXPR, 'vec4<f32>(displace(in.position, in), 1.0)');
    userVertex = '\n// --- user vertex code ---\n' + vertexCode + '\n// --- end user vertex code ---\n';
  }
  // Optional per-light / indirect hooks, wrapped so smShade can call them
  // uniformly whether or not the user supplied an override.
  let lighting = '';
  if (lightCode) {
    lighting += '\n// --- user light code ---\n' + lightCode + '\n// --- end user light code ---\n';
  }
  if (ambientCode) {
    lighting += '\n// --- user ambient code ---\n' + ambientCode + '\n// --- end user ambient code ---\n';
  }
  lighting +=
    `fn velaLight(s : Surface, l : LightSample) -> vec3<f32> { return ${lightCode ? 'light' : 'defaultLight'}(s, l); }\n` +
    `fn velaIndirect(s : Surface, ind : IndirectSample) -> vec3<f32> { return ${ambientCode ? 'ambient' : 'defaultIndirect'}(s, ind); }\n`;

  return (
    HEADER +
    SHADE_HELPERS +
    SURFACE_PRELUDE +
    uniformStruct +
    userVertex +
    vertex +
    '\n// --- user surface code ---\n' +
    surfaceCode +
    '\n// --- end user surface code ---\n' +
    lighting +
    SURFACE_SHADE
  );
}
