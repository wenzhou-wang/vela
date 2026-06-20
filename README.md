# vela

A **WebGPU-first 3D rendering engine** written from scratch in TypeScript — built to
explore what a modern engine looks like without the weight of legacy WebGL paths or
old-browser support. PBR shading, a glTF loader, and an interactive viewer in ~3k lines.

```
┌─ examples/gltf-viewer  ← drag-drop glТF viewer (the demo app)
└─ src/
   ├─ math/        Vector2/3/4, Quaternion, Matrix3/4, Euler, Color, Box3, Spherical
   ├─ core/        Object3D scene graph, Scene, Camera, PerspectiveCamera, Mesh, BufferGeometry
   ├─ geometries/  Box, Sphere, Plane primitives
   ├─ materials/   Material, StandardMaterial (PBR metallic-roughness)
   ├─ textures/    Texture (+ sampler descriptors)
   ├─ lights/      Ambient, Directional, Point
   ├─ controls/    OrbitControls (orbit / pan / dolly), FlyControls (WASD + mouse-look)
   ├─ helpers/     Grid, Axes, Box3, light gizmos (unlit lines), Stats overlay
   ├─ loaders/     GLTFLoader (.gltf + .glb), tangent generation
   └─ renderer/    WebGPURenderer, pipeline cache, geometry/texture managers,
                   mipmap generator, and the WGSL PBR shader
```

## Why WebGPU-only

three.js / Babylon / PlayCanvas carry years of WebGL1/WebGL2 compatibility code. vela
targets **WebGPU exclusively**, which means:

- A single modern shading path (WGSL), no GLSL transpilation or `#define` soup.
- Explicit GPU resources: buffers, bind groups, and pipelines are cached and reused.
- A tiny footprint — the whole engine + viewer bundles to **~90 KB (28 KB gzip)**,
  versus ~600 KB for three.js.

## Running the viewer

```bash
npm install
npm run dev        # opens the glTF viewer
```

Requires a WebGPU-capable browser (Chrome/Edge 113+, Safari 18, or Firefox with WebGPU
enabled). Then:

- **Drag & drop** a `.glb` (or a `.gltf` plus its `.bin`/textures) onto the page.
- **Open file…** picks files from disk; the sample buttons load Khronos models over the network.
- Drag to orbit · scroll to zoom · right-drag (or shift-drag) to pan.
- Sliders adjust exposure and key-light intensity; toggle auto-rotate.

Before any model is loaded it shows a **PBR material showcase**: a 5×7 grid of spheres
sweeping metalness (rows) × roughness (columns) on a checker-textured ground, lit by a
three-light rig — a quick read on whether the BRDF looks right.

## Rendering model

Forward renderer, one pass:

- **PBR metallic-roughness** BRDF — GGX NDF, Smith height-correlated visibility,
  Schlick Fresnel, Lambertian diffuse (matches the glTF 2.0 spec).
- **Multi-light** (directional + point) packed into a read-only storage buffer; ambient
  is a flat irradiance term modulated by the occlusion map.
- **Image-based lighting** — an equirectangular `scene.environment` drives mip-prefiltered
  diffuse + specular indirect light (analytic env-BRDF), replacing flat ambient when set;
  load `.hdr` panoramas via `RGBELoader` (float `DataTexture`, uploaded `rgba16float`).
- **Tangent-space normal mapping**, generated tangents when a mesh lacks them.
- **ACES filmic** tonemap + linear→sRGB encode in-shader; adjustable exposure.
- **Post-processing** (opt-in) — render to an HDR offscreen target, then fullscreen passes
  (bloom, ACES tonemap, optional FXAA) resolve to the swap chain; custom `ShaderPass`
  effects can opt into packed world-normal + linear-depth scene data.
- **Screen-space reflections** (opt-in) — HDR view-space ray marching against the shared
  hi-Z depth pyramid, with IBL-lit PBR as the fallback for misses and off-screen rays.
- **Order-independent transparency** (opt-in, weighted-blended) — accumulate + composite
  transparent fragments in the HDR pass to reduce sort artifacts.
- **4× MSAA**, depth-tested, with separate opaque / back-to-front transparent passes.
- **Directional shadow maps** — opt-in depth pass + 3×3 PCF; light frustum auto-fits the scene.
- **Instanced rendering** — `InstancedMesh` draws a whole batch in one call via a
  per-instance matrix storage buffer.
- A single **"uber" shader**: absent material maps bind white/flat 1×1 defaults, so
  pipeline variants only depend on render state (cull / blend / depth-write) plus the
  vertex path (static / skinned / instanced / morph) — not on which textures a material uses.

### Bind group layout

| Group | Contents |
|-------|----------|
| 0 | frame uniforms (view, proj, camera, ambient, shadow matrix) + lights storage + shadow map & sampler |
| 1 | per-object model + normal matrix |
| 2 | material uniforms + 5 (texture, sampler) pairs: base, normal, metal-rough, emissive, occlusion |
| 3 | bone matrices (skinned) **or** morph info + position/normal deltas + weights (morphed) |

## glTF support

- `.gltf` (external/embedded buffers & images) and binary `.glb`
- Node hierarchy with matrix or TRS transforms; multi-primitive meshes
- PBR metallic-roughness materials: base color, metallic-roughness, normal, occlusion,
  emissive textures + factors
- Per-texture samplers (wrap / filter), `KHR_materials_emissive_strength`,
  `KHR_materials_clearcoat` (factors), `KHR_materials_ior`, `KHR_materials_specular`,
  `KHR_materials_sheen`, `KHR_materials_transmission` + `_volume` (env-refraction approx)
- Vertex colors (`COLOR_0`, VEC3 or VEC4) multiplied into base color
- `OPAQUE` / `MASK` (alpha cutoff) / `BLEND` alpha modes, double-sided materials
- **Keyframe animation** (translation / rotation / scale) with STEP / LINEAR / CUBICSPLINE
  interpolation, played via `AnimationMixer`
- **GPU skinning** — `skins` / inverse bind matrices → `SkinnedMesh`, bone matrices blended
  in a skinned vertex shader variant
- **Morph targets** — `primitives[].targets` (POSITION/NORMAL deltas), default `mesh.weights`,
  `targetNames`, and `weights` animation channels; deltas blended in a morph vertex variant
- Accessor decoding with byte-stride and normalized-integer support; tangent generation
- **`EXT_meshopt_compression`** via a pluggable `MeshoptDecoder`; **`KHR_texture_basisu`/KTX2**
  via `KTX2Loader` (uncompressed RGBA8 directly; Basis through a pluggable transcoder)
- **glTF export** — `GLTFExporter` writes the scene (hierarchy, geometry, morph targets,
  skinning, PBR materials + extensions, and keyframe animation clips) back out to a binary `.glb`
- **Native scene format** — `SceneSerializer` round-trips the scene graph to/from compact
  JSON (geometry/material tables, de-duplicated)

## Verification

Because the GPU paths can't run headless, the bug-prone foundations were verified offline
(runtime-tested via esbuild + Node):

- **Math** — matrix multiply/invert, compose/decompose round-trip, WebGPU `[0,1]` depth
  for both perspective and orthographic, camera basis.
- **Frustum culling** — spheres in front / behind / beyond far / before near / off-axis;
  Gribb-Hartmann extraction for `[0,1]` clip depth.
- **Animation** — STEP / LINEAR / slerp / CUBICSPLINE interpolation, range clamping, and
  `AnimationMixer` loop-wrap.
- **Skinning** — bone = jointWorld transform, weighted multi-joint blend, and inverse-bind
  cancelling the rest pose.
- **Raycasting** — Möller–Trumbore triangle hits (front/back/miss/behind), local-space
  picking through a scaled mesh, barycentric UV interpolation, coarse-sphere fallback, and
  a CPU **BVH** whose hits match a brute-force scan across 200 rays on a 1.5k-tri mesh.
- **Debug helpers** — `GridHelper`/`AxesHelper`/`Box3Helper` geometry (vertex counts, axis
  colors, box-edge extents), light-gizmo placement/orientation/color tracking, the unlit
  line shader (bind-group + vertex-stream layout), and the `Stats` fps/ms timing core.
- **Morph targets** — `weights` track interpolation (LINEAR/STEP/CUBICSPLINE), influence
  defaults, and an end-to-end glTF load (targets, `targetNames`, weights animation).
- **Vertex colors** — all four variants parse with the added color stream at the right
  locations (skinned at 6, others at 4) with no duplicates; glTF VEC3 `COLOR_0` padded to RGBA.
- **glTF export** — a scene (transforms, geometry, indices, vertex colors, PBR materials +
  clearcoat/ior/specular/sheen, and animation clips) survives an export→`GLTFLoader` round-trip.
- **Scene format** — `SceneSerializer` round-trips hierarchy, transforms, geometry,
  lights, and materials through JSON, with shared geometry/material instances preserved.
- **Compressed assets** — meshopt buffer-view decode dispatches through a stub decoder and
  feeds accessors; KTX2 container parsing yields the right dims/format, uncompressed RGBA8
  uploads directly, and Basis routes through a transcoder hook.
- **Shadow maps** — the depth/PBR shaders parse with the shadow bindings, and the
  light-frustum fit maps the scene AABB into clip `[-1,1]²×[0,1]`.
- **Cascaded shadows** — practical uniform/log split invariants, the 288-byte four-matrix
  uniform block, atlas selection, and all PBR/ShaderMaterial variants parse offline.
- **Volumetric fog** — the clustered 3-D froxel compute shader and all material sampling
  variants parse with directional cascade shadow visibility wired into in-scattering.
- **Motion vectors** — static, skinned, instanced, and morph variants parse with previous-
  frame transforms; TAA velocity reprojection and reconstruction blur shaders parse offline.
- **Depth of field** — the depth-linearization, circle-of-confusion, and golden-angle HDR
  bokeh shader parses offline with post/MSAA prerequisite diagnostics.
- **IBL** — the frame uniform is 256 bytes with env bindings present; equirect UV mapping
  and the analytic env-BRDF fit check out at reference directions. `RGBELoader` decodes a
  hand-built `.hdr` to the right floats, and float32→float16 conversion matches known bit
  patterns.
- **Diffuse GI probes** — all material variants parse with SH-L2 grid bindings, the
  projection compute shader parses, and baked coefficients survive scene serialization.
- **Post-processing** — the fullscreen tonemap/FXAA/copy shaders parse with the expected
  bindings, and the line shader's frame struct matches the 256-byte layout.
- **WGSL** — all four vertex variants (static / skinned / instanced / morph) parsed with
  `wgsl_reflect`; bind-group indices and struct byte sizes (frame 240 / model 128 /
  material 144 / light stride 48) confirmed to match the TypeScript buffer packing.
- **ShaderMaterial lighting hooks** — the generated module parses (with the right
  vertex/fragment entry points) for every vertex variant crossed with the optional
  `light`/`ambient` hooks present or absent.
- **ShaderPass scene inputs** — StandardMaterial and ShaderMaterial modules expose both
  single-target and packed normal/linear-depth MRT entries across all vertex variants;
  the pass wrapper parses with the normal/depth helpers and five group-0 bindings.
- **SSR** — the HDR ray-march shader parses with six bindings and a 224-byte parameter
  block; the shared current-frame hi-Z path remains compatible with next-frame occlusion.
- **Shell / inverted-hull** — the per-object `Model` struct is 144 bytes with the shell
  thickness at offset 128; the PBR (static/skinned/morph), shadow, id, and Surface shaders
  all still parse with the enlarged model binding.
- **Whole project** type-checks under `strict` and bundles via Vite.

## Roadmap

The first pass is the foundation (v0.1); animation/skinning, shadows, IBL,
post-processing, compressed assets, instancing, and culling are planned. Each item has
a phase and an intended technical approach in **[ROADMAP.md](./ROADMAP.md)**.

## Documentation

- **[docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md)** — install, a first scene, loading glTF, resizing
- **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** — the frame, bind groups, the uber-shader, caching, shading model
- **[ROADMAP.md](./ROADMAP.md)** — phased plan with technical approaches

## License

MIT
