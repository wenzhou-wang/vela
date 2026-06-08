# vela roadmap

Status legend: ✅ done · 🚧 in progress · ⬜ planned

This document tracks what exists today and the planned trajectory. The guiding
constraint stays the same: **WebGPU-only, modern-first, no legacy baggage.** Each
planned item lists the intended technical approach so the work can be picked up
without re-deriving the design.

---

## v0.1 — Foundation ✅ (current)

The minimum viable engine: scene graph, PBR forward renderer, glTF viewer.

- ✅ Math library (Vec2/3/4, Quaternion, Matrix3/4, Euler, Color, Box3, Spherical)
- ✅ `Object3D` scene graph with TRS transforms and world-matrix propagation
- ✅ `PerspectiveCamera`, `Mesh`, `BufferGeometry`/`BufferAttribute`
- ✅ Box / Sphere / Plane primitives
- ✅ `StandardMaterial` — PBR metallic-roughness (glTF 2.0 model)
- ✅ Ambient / Directional / Point lights
- ✅ `WebGPURenderer` — single-pass forward, 4× MSAA, uber-shader, pipeline cache,
      geometry/texture managers, render-time mipmap generation
- ✅ `OrbitControls` with inertial damping and touch pinch-zoom
- ✅ `GLTFLoader` — `.gltf`/`.glb`, node hierarchy, PBR textures, tangent generation
- ✅ Drag-and-drop glTF viewer with exposure/light controls

---

## v0.2 — Animation & scene fidelity ⬜

Make loaded models move and sit in believable lighting.

- ✅ **Skeletal animation & skinning**
  - Keyframe animation — `KeyframeTrack` / `AnimationClip` / `AnimationMixer` with
    STEP / LINEAR / CUBICSPLINE interpolation; quaternion tracks use slerp. glTF
    `animations` (translation/rotation/scale channels) are parsed into clips; the viewer
    plays them with an animation selector.
  - GPU skinning — `Skeleton` (inverse bind matrices) + `SkinnedMesh`; bone matrices
    (`jointWorld · inverseBind`) upload to a per-mesh storage buffer (bind group 3) each
    frame, and a skinned vertex variant (JOINTS_0/WEIGHTS_0 attributes) blends them. The
    pipeline key gains a `skin` bit; the fragment stage is shared with the static path.
- ✅ **Morph targets** — `geometry.morphAttributes` (POSITION/NORMAL deltas) packed into
      storage buffers; per-mesh `morphTargetInfluences` upload to a weights storage buffer
      each frame and a morph vertex variant (bind group 3) accumulates them onto the base
      attributes (indexed by `@builtin(vertex_index)`). glTF `mesh.primitives[].targets`,
      default `mesh.weights`, `extras.targetNames`, and `weights` animation channels
      (LINEAR/STEP/CUBICSPLINE) are parsed; the pipeline key gains a `morph` variant.
- 🚧 **Shadow maps** — ✅ directional shadow mapping: a depth pass renders casters from
      the light's POV into a `depth32float` map (light frustum auto-fit to the opaque scene
      bounds), sampled with 3×3 PCF (`textureSampleCompareLevel`) in the PBR shader. Frame
      uniforms gained `lightViewProj` + shadow params; group 0 gained the shadow map +
      comparison sampler. Enable via `renderer.shadows` + `light.castShadow`. ⬜ spot/point
      shadows and a multi-light atlas.
- 🚧 **Image-based lighting (IBL)** — ✅ an equirectangular `scene.environment` drives
      indirect light: diffuse from the smallest mip, specular from a roughness-selected mip
      (the mip chain as a cheap prefilter), combined via Karis' analytic env-BRDF fit;
      replaces the flat ambient when set. Frame uniforms gained `envParams`; group 0 gained
      the env map + sampler. ✅ **`.hdr` (RGBE) loading** (`RGBELoader` → a float
      `DataTexture` uploaded as `rgba16float`). ⬜ a true GGX-prefiltered cube + irradiance
      via compute, and a precomputed BRDF LUT for higher fidelity.

---

## v0.3 — Performance & scale ⬜

Push toward "thousands of objects at 120 fps."

- ⬜ **Per-object uniform consolidation** — replace per-mesh uniform buffers with a
      single large buffer addressed by dynamic offsets (or a storage array indexed by
      `@builtin(instance_index)`), cutting bind-group churn.
- ✅ **Instanced rendering** — `InstancedMesh` stores per-instance model matrices in a
      storage buffer read by an instanced vertex variant (indexed by `instance_index`);
      the whole batch draws in one `drawIndexed(..., count)` call. The viewer's
      "Instances ✦" demo renders a 1,600-cube field as a single draw.
- ✅ **Frustum culling** — `Frustum`/`Plane`/`Sphere` math against world-space bounding
      spheres; off-screen meshes skipped on the CPU each frame. Toggle via
      `renderer.frustumCulling`, per-object opt-out via `object.frustumCulled`, and
      `renderer.culledCount` reports the last frame's skipped meshes.
- ⬜ **GPU-driven culling** — compute-shader frustum/occlusion cull writing an indirect
      draw buffer (`drawIndexedIndirect`).
- ✅ **Render bundles** — opt-in `renderer.renderBundles` records the opaque draws into a
      `GPURenderBundle` and replays it, re-recording only when the draw set (or frame bind
      group) changes; per-object uniform buffers are still refreshed each frame so dynamic
      transforms stay correct.

---

## v0.4 — Visual quality ⬜

A post-processing stack and richer materials.

- ✅ **Render graph / post pipeline** — opt-in `renderer.postProcessing`: the scene renders
      to an HDR (`rgba16float`) offscreen target (MSAA-resolved), then a chain of fullscreen
      triangle passes resolves to the swap chain. The material/line shaders skip their
      in-shader tonemap when a "linear output" frame flag is set.
- 🚧 **Post effects** — ✅ a tone-mapping pass (ACES moved out of the material shader),
      ✅ **FXAA** (`renderer.fxaa`), and ✅ **bloom** (`renderer.bloom`: bright-pass +
      half-res separable-Gaussian blur, added before tonemap). ⬜ SSAO, optional TAA with
      motion vectors.
- 🚧 **Material extensions** — ✅ **vertex colors** (glTF `COLOR_0`, VEC3/VEC4): an
      always-present per-vertex color stream (white default) multiplies base color in every
      vertex variant, so no new pipeline is needed. ✅ **`KHR_materials_clearcoat`** (factors):
      a second GGX specular lobe over the base, attenuating it by the coat's Fresnel; the
      `clearcoat`/`clearcoatRoughness` factors reuse spare material-uniform slots (no size
      change). ✅ **`KHR_materials_ior`** + **`KHR_materials_specular`** shape the dielectric
      F0 (IOR → F0, scaled/tinted by the specular factor & color); defaults (ior 1.5, white,
      1) reproduce the old flat 0.04. ✅ **`KHR_materials_sheen`** adds a Charlie-NDF
      retroreflective lobe for cloth (black sheen color = disabled). ✅ **`KHR_materials_transmission`**
      + **`_volume`** refract the environment/ambient through the surface with Beer–Lambert
      volume attenuation (an approximation avoiding a screen-space capture). ⬜ clearcoat
      textures, true screen-space refraction.
- ✅ **Transparency (weighted-blended OIT)** — opt-in `renderer.oit` (requires the HDR post
      path + sampleCount 1): transparent meshes accumulate weighted premultiplied color into
      an `accum` target and product-of-(1−α) into a `reveal` target (depth-tested, no write),
      then a fullscreen pass composites them onto the HDR scene before tonemapping. A shared
      `shadeSurface()` feeds both the opaque `fs_main` and the OIT `fs_oit`. ⬜ MSAA support.

---

## v0.5 — Assets & ecosystem ⬜

Faster loads, broader inputs, and ergonomics.

- 🚧 **Compressed geometry** — ✅ **`EXT_meshopt_compression`**: the loader decodes
      meshopt buffer views through a pluggable `MeshoptDecoder` (`loader.setMeshoptDecoder(MeshoptDecoder)`,
      the standard meshoptimizer module), then accessors read the decoded data. ⬜ `KHR_draco_mesh_compression`.
- 🚧 **Compressed textures** — ✅ **`KHR_texture_basisu` / KTX2**: `KTX2Loader` parses the
      container, uploads uncompressed RGBA8 directly, and transcodes Basis (ETC1S/UASTC) via
      a pluggable transcoder that yields RGBA8 (`loader.setKTX2Loader(new KTX2Loader().setTranscoder(...))`).
      ⬜ transcode straight to the platform's preferred GPU-compressed format (BC7/ASTC/ETC2).
- ⬜ **Worker-based loading** — parse glTF and decode images off the main thread;
      transfer typed arrays / `ImageBitmap`s.
- ✅ **More cameras/controls** — `OrthographicCamera`; `FlyControls` (WASD + Q/E,
      mouse-look via drag or pointer lock, frame-rate-independent `update(delta)`).
- ✅ **Raycasting / picking** — `Ray` + `Raycaster` pick meshes; the broad phase
      rejects on the world bounding sphere, then survivors are tested per-triangle in
      local space (Möller–Trumbore), so hits carry an exact `point`, `faceIndex`, and a
      barycentric-interpolated `uv`. `setFromCamera` unprojects NDC; `precise = false`
      reverts to the coarse sphere test. Meshes at/above `bvhThreshold` triangles use a
      cached median-split **CPU `BVH`** that prunes subtrees the ray misses. ⬜ A GPU
      id-buffer pass for pixel-exact picking remains optional.
- ✅ **Helpers & debug** — `GridHelper`, `AxesHelper`, `Box3Helper`, and light gizmos
      (`DirectionalLightHelper`, `PointLightHelper`) via an unlit `LineSegments` +
      `LineBasicMaterial` path (line-list pipeline, per-vertex colors); plus a `Stats`
      FPS/ms overlay with a rolling graph (DOM-decoupled timing core).

---

## Backlog / under consideration ⬜

- Optional **WebGL2 fallback** behind the same scene-graph API (only if demand warrants
  the cost — it cuts against the project's modern-first thesis).
- **Clustered forward+** lighting for hundreds of dynamic lights.
- ✅ **glTF export** (`GLTFExporter`) — writes node hierarchy + TRS, mesh geometry
  (position/normal/uv/color/indices), and `StandardMaterial` PBR factors with the
  clearcoat/ior/specular/sheen extensions, **morph targets** (deltas + default weights +
  `targetNames`), and keyframe **animation** clips (translation/rotation/scale/weights, all
  interpolations) and **skinning** (JOINTS/WEIGHTS, skins + inverse bind matrices) to a
  binary `.glb`; verified by round-tripping back through `GLTFLoader`. ⬜ textures.
- ✅ **Lightweight scene format** (`SceneSerializer`) — lossless JSON round-trip of the scene
  graph (`Object3D`/`Mesh`/`LineSegments`/lights, transforms, `BufferGeometry` attributes,
  `StandardMaterial`/`LineBasicMaterial`), with geometry/material de-duplication preserved
  across load. Verified by a serialize→stringify→parse→deserialize round-trip.
- **WebXR** session support.
- A small **node-based material** graph compiling to WGSL.

---

## Design principles (don't regress these)

1. **One modern path.** No `#ifdef`-style capability forks; target WebGPU's baseline.
2. **Cache aggressively, allocate rarely.** Pipelines, bind groups, and GPU buffers are
   created once and reused; the hot render loop should not allocate.
3. **Pipeline variants are earned.** A new pipeline key must reflect real render-state
   divergence — keep leaning on the uber-shader + default-texture approach.
4. **Verifiable offline.** Math and shader-interface invariants (bind-group indices,
   struct sizes) stay testable without a GPU. See the Verification section of the README.
5. **Stay lean.** Track bundle size; a feature that doubles the footprint needs to earn it.
