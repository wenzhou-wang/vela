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
   ├─ helpers/     GridHelper, AxesHelper, Box3Helper (unlit line gizmos)
   ├─ loaders/     GLTFLoader (.gltf + .glb), tangent generation
   └─ renderer/    WebGPURenderer, pipeline cache, geometry/texture managers,
                   mipmap generator, and the WGSL PBR shader
```

## Why WebGPU-only

three.js / Babylon / PlayCanvas carry years of WebGL1/WebGL2 compatibility code. vela
targets **WebGPU exclusively**, which means:

- A single modern shading path (WGSL), no GLSL transpilation or `#define` soup.
- Explicit GPU resources: buffers, bind groups, and pipelines are cached and reused.
- A tiny footprint — the whole engine + viewer bundles to **~86 KB (26 KB gzip)**,
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
- **Tangent-space normal mapping**, generated tangents when a mesh lacks them.
- **ACES filmic** tonemap + linear→sRGB encode in-shader; adjustable exposure.
- **4× MSAA**, depth-tested, with separate opaque / back-to-front transparent passes.
- **Instanced rendering** — `InstancedMesh` draws a whole batch in one call via a
  per-instance matrix storage buffer.
- A single **"uber" shader**: absent material maps bind white/flat 1×1 defaults, so
  pipeline variants only depend on render state (cull / blend / depth-write) plus the
  vertex path (static / skinned / instanced / morph) — not on which textures a material uses.

### Bind group layout

| Group | Contents |
|-------|----------|
| 0 | frame uniforms (view, proj, camera, ambient) + lights storage buffer |
| 1 | per-object model + normal matrix |
| 2 | material uniforms + 5 (texture, sampler) pairs: base, normal, metal-rough, emissive, occlusion |
| 3 | bone matrices (skinned) **or** morph info + position/normal deltas + weights (morphed) |

## glTF support

- `.gltf` (external/embedded buffers & images) and binary `.glb`
- Node hierarchy with matrix or TRS transforms; multi-primitive meshes
- PBR metallic-roughness materials: base color, metallic-roughness, normal, occlusion,
  emissive textures + factors
- Per-texture samplers (wrap / filter), `KHR_materials_emissive_strength`
- `OPAQUE` / `MASK` (alpha cutoff) / `BLEND` alpha modes, double-sided materials
- **Keyframe animation** (translation / rotation / scale) with STEP / LINEAR / CUBICSPLINE
  interpolation, played via `AnimationMixer`
- **GPU skinning** — `skins` / inverse bind matrices → `SkinnedMesh`, bone matrices blended
  in a skinned vertex shader variant
- **Morph targets** — `primitives[].targets` (POSITION/NORMAL deltas), default `mesh.weights`,
  `targetNames`, and `weights` animation channels; deltas blended in a morph vertex variant
- Accessor decoding with byte-stride and normalized-integer support; tangent generation

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
  colors, box-edge extents) and the unlit line shader (bind-group + vertex-stream layout).
- **Morph targets** — `weights` track interpolation (LINEAR/STEP/CUBICSPLINE), influence
  defaults, and an end-to-end glTF load (targets, `targetNames`, weights animation).
- **WGSL** — all four vertex variants (static / skinned / instanced / morph) parsed with
  `wgsl_reflect`; bind-group indices and struct byte sizes (frame 160 / model 128 /
  material 64 / light stride 48) confirmed to match the TypeScript buffer packing.
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
