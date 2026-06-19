# ShaderMaterial — custom surfaces in one WGSL function

`ShaderMaterial` lets you write a single WGSL function that produces PBR surface
inputs. The engine supplies everything else: all light types, shadows, IBL,
clustered lighting, OIT transparency, post-processing and tonemapping behave
exactly as they do for `StandardMaterial`.

## Minimal example

```ts
import { ShaderMaterial, Mesh, BoxGeometry } from 'vela';

const material = new ShaderMaterial({
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      s.baseColor = vec3(in.uv, 1.0);
      return s;
    }
  `,
});
const mesh = new Mesh(new BoxGeometry(), material);
```

That is the whole contract: define `fn surface(in : VSOut) -> Surface`, start
from `defaultSurface(in)`, override what you need, return it.

## The Surface struct

| Field | Type | Default | Meaning |
|---|---|---|---|
| `baseColor` | `vec3<f32>` | `vec3(0.8)` | Albedo, linear color |
| `alpha` | `f32` | `1.0` | Opacity (set `transparent: true` on the material) |
| `metalness` | `f32` | `0.0` | 0 = dielectric, 1 = metal |
| `roughness` | `f32` | `0.5` | Perceptual roughness, clamped to 0.04..1 |
| `emissive` | `vec3<f32>` | `vec3(0.0)` | Emitted light, linear, HDR values welcome |
| `normal` | `vec3<f32>` | `in.worldNormal` | World-space shading normal |
| `occlusion` | `f32` | `1.0` | Ambient/IBL occlusion factor |

## What your function can read

- **`in : VSOut`** — interpolated vertex data:
  `in.worldPos` (vec3), `in.worldNormal` (vec3), `in.uv` (vec2),
  `in.worldTangent` (vec3), `in.tangentSign` (f32), `in.color` (vec4, vertex
  color, white when absent), `in.clipPosition` (vec4, pixel position).
- **`u.<name>`** — your custom uniforms (see below).
- **`elapsedTime()`** — seconds since the renderer started; drive animation with it.
- **`frame`** — engine frame data (`frame.cameraPos.xyz`, `frame.view`, `frame.proj`, ...).

## Custom uniforms

Declare uniforms as a plain object; they appear in WGSL as `u.<name>` with the
matching type. Mutate values freely — they upload every frame, no dirty flags:

| JS value | WGSL |
|---|---|
| `number` | `u.<name> : f32` |
| `Vector2` | `u.<name> : vec2<f32>` |
| `Vector3`, `Color` | `u.<name> : vec3<f32>` |
| `Vector4` | `u.<name> : vec4<f32>` |
| `Texture` | `t_<name> : texture_2d<f32>` + `s_<name> : sampler` |

```ts
import { ShaderMaterial, Color } from 'vela';

const lava = new ShaderMaterial({
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      let pulse = 0.5 + 0.5 * sin(elapsedTime() * u.speed + in.worldPos.y * 3.0);
      s.baseColor = vec3(0.05);
      s.roughness = 0.9;
      s.emissive = u.glowColor * pulse * u.intensity;
      return s;
    }
  `,
  uniforms: {
    speed: 2.0,
    intensity: 3.0,
    glowColor: new Color(1.0, 0.3, 0.05),
  },
});

// later, anywhere:
lava.uniforms.intensity = 5.0;            // applied next frame
lava.uniforms.glowColor = new Color(0, 1, 0);
```

Adding or removing a uniform key (or changing a value's type) recompiles the
shader automatically on the next frame.

## Texture uniforms

A `Texture` value becomes a `t_<name>` / `s_<name>` texture+sampler pair you
sample directly. Swapping the texture (or bumping its `version`) rebinds it on
the next frame:

```ts
import { ShaderMaterial, Vector2, TextureLoader } from 'vela';

const albedo = await new TextureLoader().load('brick.png');
const mat = new ShaderMaterial({
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      s.baseColor = textureSample(t_albedo, s_albedo, in.uv * u.tiling).rgb;
      s.roughness = textureSample(t_rough, s_rough, in.uv).r;
      return s;
    }
  `,
  uniforms: { tiling: new Vector2(4, 4), albedo, rough: roughnessTex },
});
```

## Vertex displacement

An optional `vertex` function `displace(position, in) -> vec3<f32>` runs in the
vertex stage before the model transform — for waves, wind, inflation, terrain.
It applies to plain and instanced meshes (skinned/morph meshes ignore it). The
shading normal isn't recomputed automatically, so perturb `s.normal` in
`surface()` if the lighting should follow the displacement.

```ts
const flag = new ShaderMaterial({
  uniforms: { amp: 0.15 },
  vertex: /* wgsl */ `
    fn displace(position : vec3<f32>, in : VSIn) -> vec3<f32> {
      let wave = sin(position.x * 6.0 + elapsedTime() * 3.0) * u.amp;
      return position + vec3(0.0, 0.0, wave);
    }
  `,
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      s.baseColor = vec3(0.8, 0.1, 0.1);
      return s;
    }
  `,
});
// swap it live:
flag.setVertex(null);     // back to undisplaced
```

`VSIn` exposes `in.position`, `in.normal`, `in.uv`, `in.tangent`, `in.color`.

## Custom lighting

`surface()` produces material inputs; the engine then runs its PBR lighting over
them. Two optional hooks let you intercept that lighting instead of re-deriving
it — the engine still decides *which* lights reach the fragment (clustered list,
distance/cone attenuation, shadow visibility) and hands you the per-light terms.
This is how toon ramps, wrap / half-Lambert, and posterized highlights are
expressed without an engine flag.

`light(s, l) -> vec3<f32>` is called once per reaching light; return that light's
contribution and the engine sums them. `LightSample` carries:

| field        | meaning                                                      |
| ------------ | ----------------------------------------------------------- |
| `l.L`        | world-space direction to the light                          |
| `l.radiance` | light color × distance/cone attenuation × **shadow** factor |
| `l.N` `l.V` `l.H` | shading normal, view dir, half vector                  |
| `l.NoL`      | **raw** `dot(N, L)` — may be negative (for wrap lighting)   |
| `l.NoV` `l.NoH` `l.VoH` | the usual clamped dots for BRDF terms            |

`ambient(s, ind) -> vec3<f32>` replaces the indirect (image-based / flat ambient)
term; `IndirectSample` carries `ind.N`, `ind.V`, `ind.NoV`.

Call `defaultLight(s, l)` / `defaultIndirect(s, ind)` to reuse (or blend with)
the engine's PBR result.

```ts
const toon = new ShaderMaterial({
  uniforms: { ramp: new Vector3(0.0, 0.5, 1.0) },
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      s.baseColor = vec3(0.85, 0.2, 0.25);
      return s;
    }
  `,
  // Two-band cel ramp on N·L, with a hard specular dot.
  light: /* wgsl */ `
    fn light(s : Surface, l : LightSample) -> vec3<f32> {
      let band = select(u.ramp.x, select(u.ramp.y, u.ramp.z, l.NoL > 0.66), l.NoL > 0.33);
      let spec = select(0.0, 1.0, l.NoH > 0.98);
      return (s.baseColor * band + vec3(spec)) * l.radiance;
    }
  `,
  // Flat fill instead of full IBL.
  ambient: /* wgsl */ `
    fn ambient(s : Surface, ind : IndirectSample) -> vec3<f32> {
      return s.baseColor * 0.15;
    }
  `,
});
toon.setLight(null);   // back to physical direct lighting
toon.setAmbient(null); // back to the engine's indirect
```

Wrap / half-Lambert lighting uses the raw `l.NoL`:

```wgsl
fn light(s : Surface, l : LightSample) -> vec3<f32> {
  let wrap = clamp((l.NoL + u.wrap) / (1.0 + u.wrap), 0.0, 1.0);
  return s.baseColor * wrap * l.radiance;
}
```

## More recipes

**Procedural stripes with a custom normal wobble:**

```ts
const candy = new ShaderMaterial({
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      let stripe = step(0.5, fract(in.uv.x * u.count));
      s.baseColor = mix(vec3(0.9, 0.1, 0.2), vec3(0.95), stripe);
      s.roughness = mix(0.15, 0.6, stripe);
      s.normal = normalize(in.worldNormal + 0.08 * sin(in.worldPos * 40.0));
      return s;
    }
  `,
  uniforms: { count: 8.0 },
});
```

**Transparent force field (works with `renderer.oit`):**

```ts
const field = new ShaderMaterial({
  transparent: true,
  side: 'double',
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      let fresnel = pow(1.0 - abs(dot(normalize(in.worldNormal),
        normalize(frame.cameraPos.xyz - in.worldPos))), 3.0);
      s.baseColor = vec3(0.1, 0.5, 1.0);
      s.emissive = vec3(0.2, 0.7, 1.5) * fresnel;
      s.alpha = 0.15 + 0.6 * fresnel;
      return s;
    }
  `,
});
```

**Hot-swapping the shader (for live iteration):**

```ts
material.setSurface(newWgslSource); // recompiles on next draw
```

## Error handling

- Constructing a `ShaderMaterial` whose `surface` doesn't define
  `fn surface(` throws immediately with the expected signature.
- Invalid uniform names or value types throw with the accepted types listed.
- WGSL compile errors are reported to the console with the failing line of the
  *generated* module and its line number, e.g.
  `[vela] ShaderMaterial "lava" failed to compile: :312:18 unresolved value 'glowColr'`.

## What works everywhere

ShaderMaterial meshes participate in: shadows (as casters), frustum + GPU
culling, instancing (`InstancedMesh`), skinning (`SkinnedMesh`), morph targets,
OIT, TAA, SSAO, bloom, picking (`renderer.pickAt`), and render bundles.
Surface functions run identically with and without `renderer.postProcessing`.

## Limitations

- Texture uniforms are 2D only (no cube/array/3D textures yet).
- The vertex hook applies to plain and instanced meshes; skinned and
  morph-target meshes use their built-in vertex stages unchanged.
- Transmission/clearcoat/sheen lobes are `StandardMaterial`-only.

---

# ShaderPass — custom post effects

`ShaderPass` is the screen-space sibling of `ShaderMaterial`: you write one
WGSL function over the full frame, and it runs as a post-processing stage.
Passes run in order in **HDR linear space**, after bloom/SSAO/TAA and before
tonemapping, so effects compose correctly with the rest of the pipeline.
Requires `renderer.postProcessing = true`.

```ts
import { ShaderPass } from 'vela';

renderer.postProcessing = true;

// Heat-haze wobble.
renderer.passes.push(new ShaderPass({
  effect: /* wgsl */ `
    fn effect(uv : vec2<f32>) -> vec4<f32> {
      let wobble = sin(uv.y * 80.0 + pp.time * 4.0) * u.amount;
      return sceneColor(uv + vec2(wobble, 0.0));
    }
  `,
  uniforms: { amount: 0.008 },
}));
```

## The effect function

Define `fn effect(uv : vec2<f32>) -> vec4<f32>` returning the output color for
that pixel (`uv` is 0..1, origin top-left). In scope:

- `sceneColor(uv)` — the previous stage's color (HDR linear). Also `sceneTex` +
  `sceneSmp` if you want manual sampling.
- `sceneDepth(uv)` — non-linear depth in `[0,1]` (1 = background), for
  depth-aware effects like fog or DoF.
- `sceneWorldNormal(uv)` / `sceneViewNormal(uv)` — the scene's shading normal,
  when the pass requests the `normal` input.
- `sceneLinearDepth(uv)` — positive view-space distance in world units (the
  camera far plane for background), when the pass requests `linearDepth`.
- `pp.resolution` — `vec4` (xy = pixels, zw = 1/pixels); `pp.time` — seconds.
- `u.<name>` / `t_<name>` + `s_<name>` — your `uniforms`, same types and rules
  as ShaderMaterial.

## Managing the chain

```ts
const grade = new ShaderPass({ effect: `...`, uniforms: { exposure: 1.2 } });
renderer.passes.push(grade);

grade.uniforms.exposure = 0.8; // applied next frame
grade.enabled = false;         // skip without removing
grade.setEffect(newWgsl);      // recompiles next frame
renderer.passes = [];          // clear all
```

Passes execute in array order; reorder the array to reorder effects. Like
ShaderMaterial, an invalid `effect` throws on construction, and WGSL compile
errors print the failing generated line to the console.

## Normal and linear-depth inputs

Request geometry-aware inputs explicitly:

```ts
const edges = new ShaderPass({
  inputs: ['normal', 'linearDepth'],
  effect: /* wgsl */ `
    fn effect(uv : vec2<f32>) -> vec4<f32> {
      let px = pp.resolution.zw;
      let normalEdge = length(sceneViewNormal(uv) - sceneViewNormal(uv + vec2(px.x, 0.0)));
      let depthEdge = abs(sceneLinearDepth(uv) - sceneLinearDepth(uv + vec2(0.0, px.y)));
      let edge = clamp(max(normalEdge, depthEdge * 0.1), 0.0, 1.0);
      return vec4(sceneColor(uv).rgb * (1.0 - edge), 1.0);
    }
  `,
});
```

When any enabled pass requests one of these inputs, the scene pass writes one
additional RGBA16F target: RGB stores the world-space shading normal and alpha
stores normalized linear depth. Both helpers are available to every pass in
that frame. Without a request, the renderer keeps the original single-target
scene path and incurs no attachment bandwidth.
