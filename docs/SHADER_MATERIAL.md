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

| JS value | WGSL type |
|---|---|
| `number` | `f32` |
| `Vector2` | `vec2<f32>` |
| `Vector3`, `Color` | `vec3<f32>` |
| `Vector4` | `vec4<f32>` |

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

## Limitations (v1)

- No custom textures yet — uniforms only. Use `StandardMaterial` for textured
  surfaces, or bake data into vertex colors/UVs.
- No custom vertex stage; the four built-in vertex variants are used as-is.
- Transmission/clearcoat/sheen lobes are `StandardMaterial`-only.
