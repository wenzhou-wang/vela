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
- ⬜ **Morph targets** — POSITION/NORMAL deltas as storage buffers, weights in the
      model uniform; resolve in the vertex shader.
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
- ⬜ **Instanced rendering** — `InstancedMesh` with a per-instance transform storage
      buffer; one draw call per geometry/material.
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
- ⬜ **Material extensions** — `KHR_materials_clearcoat`, `_transmission`/`_volume`
      (refraction via screen-space sampling), `_sheen`, `_specular`, `_ior`, vertex colors.
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
- ⬜ **More cameras/controls** — `OrthographicCamera`, first-person & fly controls.
- ⬜ **Raycasting / picking** — CPU BVH for interaction, or a GPU id-buffer pass.
- ⬜ **Helpers & debug** — grid, axes, bounding-box and light gizmos; a stats/inspector overlay.

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
