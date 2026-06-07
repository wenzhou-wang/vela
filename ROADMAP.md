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
- ⬜ **Shadow maps** — directional/spot depth pass into a depth atlas; PCF (3×3) in the
      PBR shader. Adds a `lightViewProj` array to the frame uniforms and a shadow
      sampler/atlas to group 0.
- ⬜ **Image-based lighting (IBL)** — load HDR/`.hdr` equirect, compute a diffuse
      irradiance cube and a GGX-prefiltered specular mip chain (compute shaders) +
      a precomputed BRDF LUT. Replaces the flat ambient term with proper indirect light.

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
- ⬜ **Render bundles** — record static opaque draws into `GPURenderBundle`s to amortize
      encoding cost.

---

## v0.4 — Visual quality ⬜

A post-processing stack and richer materials.

- ⬜ **Render graph / post pipeline** — render scene to an HDR (`rgba16float`) offscreen
      target, then a chain of fullscreen passes resolving to the swap chain.
- ⬜ **Post effects** — bloom (threshold + Kawase blur), SSAO, tone-mapping pass (move
      ACES out of the material shader), FXAA, optional TAA with motion vectors.
- 🚧 **Material extensions** — ✅ **vertex colors** (glTF `COLOR_0`, VEC3/VEC4): an
      always-present per-vertex color stream (white default) multiplies base color in every
      vertex variant, so no new pipeline is needed. ✅ **`KHR_materials_clearcoat`** (factors):
      a second GGX specular lobe over the base, attenuating it by the coat's Fresnel; the
      `clearcoat`/`clearcoatRoughness` factors reuse spare material-uniform slots (no size
      change). ⬜ clearcoat textures, `_transmission`/`_volume` (screen-space refraction),
      `_sheen`, `_specular`, `_ior`.
- ⬜ **Transparency** — optional weighted-blended OIT to reduce sort artifacts.

---

## v0.5 — Assets & ecosystem ⬜

Faster loads, broader inputs, and ergonomics.

- ⬜ **Compressed geometry** — `KHR_draco_mesh_compression` and `EXT_meshopt_compression`
      (WASM decoders in a Worker).
- ⬜ **Compressed textures** — `KTX2`/Basis Universal transcoding to the platform's
      preferred GPU format.
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
- 🚧 **Helpers & debug** — ✅ `GridHelper`, `AxesHelper`, `Box3Helper`, and light gizmos
      (`DirectionalLightHelper`, `PointLightHelper`) via an unlit `LineSegments` +
      `LineBasicMaterial` path (line-list pipeline, per-vertex colors); ⬜ a stats/inspector overlay.

---

## Backlog / under consideration ⬜

- Optional **WebGL2 fallback** behind the same scene-graph API (only if demand warrants
  the cost — it cuts against the project's modern-first thesis).
- **Clustered forward+** lighting for hundreds of dynamic lights.
- **glTF export** and a lightweight scene format.
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
