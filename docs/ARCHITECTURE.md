# Architecture

How vela turns a scene graph into pixels. This is the mental model you need to extend the
renderer; the [ROADMAP](../ROADMAP.md) builds on these structures.

## Layers

```
 Scene graph (CPU)            Renderer (GPU orchestration)         WebGPU
 ─────────────────            ────────────────────────────        ──────
 Object3D ─ Mesh              WebGPURenderer.render()              GPUDevice
   ├ geometry: BufferGeometry   ├ collect()  → opaque/transparent  GPUBuffer
   ├ material: StandardMaterial ├ uploadFrame() → frame+lights     GPUTexture
   └ matrixWorld                ├ PipelineCache                    GPURenderPipeline
 Camera, Light                  ├ GeometryBuffers (WeakMap cache)  GPUBindGroup
                                ├ TextureManager  (+ mipmaps)      WGSL (pbr shader)
                                └ per-mesh / per-material caches
```

Everything above the renderer is plain data and math — no GPU types leak into the scene
graph. The renderer is the only place that talks to WebGPU.

## The frame

`renderer.render(scene, camera)` does, in order:

1. **Update transforms** — `scene.updateMatrixWorld()` propagates local TRS into world
   matrices down the tree; the camera computes its `matrixWorldInverse` (the view matrix).
2. **Collect** — one `traverseVisible` pass buckets meshes into **opaque** and
   **transparent** lists and gathers **lights**.
3. **Upload frame state** — view/projection/camera-position/ambient go into a 160-byte
   uniform buffer; each light is packed into a 48-byte stride in a read-only storage buffer.
   Ambient lights are folded into the flat ambient term rather than the light array.
4. **Sort** transparent meshes back-to-front by distance to the camera.
5. **Encode one render pass** — bind frame state (group 0) once, then for each mesh set its
   pipeline, model bind group (group 1), material bind group (group 2), vertex/index
   buffers, and issue the draw. Opaque first, then transparent.
6. **Submit** the command buffer.

MSAA (4× by default) renders into a multisampled color target and resolves into the swap
chain texture; a matching multisampled depth texture is used for depth testing.

## Bind group layout

Three bind groups, fixed across all pipelines so they bind once and stay stable:

| Group | Binding | Resource | Visibility |
|-------|---------|----------|------------|
| **0** frame | 0 | `Frame` uniform: view, proj, camera+numLights, ambient+exposure | vertex + fragment |
|        | 1 | `array<Light>` read-only storage | fragment |
| **1** model | 0 | `Model` uniform: model matrix + normal matrix | vertex |
| **2** material | 0 | `MaterialU` uniform: base color, emissive, metal/rough/normal/AO factors | fragment |
|        | 1–10 | 5 × (texture, sampler): base, normal, metal-rough, emissive, occlusion | fragment |

Struct byte sizes (frame 160 / model 128 / material 64 / light 48) are asserted against the
TypeScript buffer-packing code via `wgsl_reflect` — a mismatch there silently corrupts
rendering, so it's verified offline.

## The "uber shader" and pipeline variants

There is **one** WGSL shader (`src/renderer/shaders/pbr.wgsl.ts`). Materials that lack a
given texture bind a shared **1×1 default** instead — white for color/metal-rough/emissive/AO
(so a multiply is a no-op) and flat (0.5, 0.5, 1) for normals (so the TBN resolves to the
geometric normal even with placeholder tangents).

The payoff: **pipeline variants don't depend on which textures a material has.** A pipeline
is keyed only by real render-state divergence:

```
cullMode (front|back|none) × blend (opaque|blend) × depthWrite (on|off)
```

So a scene with 200 differently-textured materials still compiles only a handful of
pipelines. `PipelineCache` memoizes them by that key.

## Resource caching

- **`GeometryBuffers`** — `WeakMap<BufferGeometry, GPUBuffers>`. Lazily uploads position/
  normal/uv/tangent (filling defaults for missing streams so the vertex layout is fixed) and
  the index buffer. Re-uploads when the geometry's `version` changes.
- **`TextureManager`** — `WeakMap<Texture, {texture, view, sampler}>`. Uploads via
  `copyExternalImageToTexture`, generates mipmaps, and owns the shared default textures and
  sampler. sRGB vs linear is chosen from the texture's `colorSpace`.
- **`MipmapGenerator`** — builds the mip chain with a fullscreen-triangle blit per level;
  pipelines cached per format.
- **Per-mesh / per-material** — small uniform buffers + bind groups cached on the object,
  updated each frame (cheap) and rebuilt only when the texture set changes.

## Shading model

PBR metallic-roughness, matching the glTF 2.0 spec:

- **Specular**: Cook-Torrance — GGX normal distribution, Smith height-correlated visibility,
  Schlick Fresnel (F0 = 0.04 for dielectrics, tinted by base color for metals).
- **Diffuse**: Lambertian, energy-conserved against the Fresnel term, zeroed for metals.
- **Lights**: directional (parallel) and point (inverse-square with optional range falloff).
- **Ambient**: a flat irradiance term modulated by the occlusion map (this is the seam where
  IBL will plug in — see the roadmap).
- **Output**: emissive added, exposure applied, **ACES filmic** tonemap, then **linear→sRGB**
  encode (the swap chain is a non-sRGB format, so encoding happens in-shader).

Coordinate system is right-handed, Y-up, with WebGPU's `[0, 1]` clip-space depth — the
projection matrix and the depth compare (`less`, clear to 1.0) are built for that.

## Where things live

```
src/renderer/
  WebGPURenderer.ts    frame orchestration, render pass, per-object/material caches
  PipelineCache.ts     bind group layouts + pipeline compilation/memoization
  GeometryBuffers.ts   geometry → GPU vertex/index buffers
  TextureManager.ts    textures → GPU textures/samplers, default resources
  MipmapGenerator.ts   blit-based mip chain
  constants.ts         vertex layout, formats
  shaders/pbr.wgsl.ts  the WGSL shader (as a TS template string)
```
