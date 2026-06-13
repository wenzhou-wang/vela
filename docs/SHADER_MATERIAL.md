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
