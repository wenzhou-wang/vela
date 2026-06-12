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

## v0.2 — Animation & scene fidelity ✅

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
- ✅ **Shadow maps** — ✅ directional shadow mapping: a depth pass renders casters from
      the light's POV into a `depth32float` map (light frustum auto-fit to the opaque scene
      bounds), sampled with 3×3 PCF (`textureSampleCompareLevel`) in the PBR shader. Frame
      uniforms gained `lightViewProj` + shadow params; group 0 gained the shadow map +
      comparison sampler. Enable via `renderer.shadows` + `light.castShadow`. ✅ **Spot
      lights + shadow atlas**: `SpotLight` (angle, penumbra, distance, decay, castShadow,
      target) packs into the light storage buffer with kind=2; angular attenuation uses inner/
      outer cosine thresholds with squared smoothstep. Shadow-casting spot lights (up to 4)
      each write a 512×512 tile in a 2048×2048 depth atlas via `setViewport`; per-tile
      `viewProj` matrices and UV-offset/scale region descriptors are uploaded in a 320-byte
      storage buffer (binding 10); the PBR fragment shader does 3×3 PCF against the correct
      atlas tile per spot light. ✅ **Point-light cube shadow atlas**: shadow-casting
      point lights (up to 2, via `pointLight.castShadow`) render 6 cube faces (90°
      perspective, far = `distance` or 200) into 6 consecutive tiles of the same
      2048×2048 atlas (after the 4 spot tiles — 4+12 = all 16 tiles); the PBR shader
      picks the face from the dominant axis of the light→fragment vector and reuses
      the spot-tile 3×3 PCF path.
- ✅ **Image-based lighting (IBL)** — an equirectangular `scene.environment` drives
      indirect light. ✅ Initial path: diffuse from smallest mip, specular from
      roughness-scaled LOD, Karis analytic env-BRDF fit. ✅ **`.hdr` (RGBE) loading**
      (`RGBELoader` → a float `DataTexture` uploaded as `rgba16float`). ✅ **GGX-prefiltered
      IBL + BRDF LUT**: `IBLPrefilter` runs three compute passes when `scene.environment`
      changes — a 64-sample cosine-hemisphere irradiance convolution (64×32 RGBA16F), a
      128-sample GGX importance-sampling specular prefilter with 6 roughness-mip levels
      (256×128 RGBA16F), and a 512-sample split-sum BRDF LUT (128×128 RGBA16F, computed
      once at startup via Hammersley + Smith GGX geometry).  Frame bindings 6–9 carry the
      irradiance map and BRDF LUT; envParams.w bit 1 activates the high-fidelity path in
      the PBR shader; envParams.w bit 0 retains the existing linear-output flag.

---

## v0.3 — Performance & scale ✅

Push toward "thousands of objects at 120 fps."

- ✅ **Per-object uniform consolidation** — a single `GPUBuffer` pool (256-byte-aligned
      slots, one per mesh, lazily grown × 2) replaces per-mesh `GPUBuffer` allocations;
      the model bind group layout uses `hasDynamicOffset: true`, so all opaque/transparent/
      shadow/line/id draw calls share one `modelPoolBindGroup` and switch models via
      `setBindGroup(1, pool, [slot * 256])` — cutting GC pressure and GPU-side bind-group
      object count proportional to scene size.
- ✅ **Instanced rendering** — `InstancedMesh` stores per-instance model matrices in a
      storage buffer read by an instanced vertex variant (indexed by `instance_index`);
      the whole batch draws in one `drawIndexed(..., count)` call. The viewer's
      "Instances ✦" demo renders a 1,600-cube field as a single draw.
- ✅ **Frustum culling** — `Frustum`/`Plane`/`Sphere` math against world-space bounding
      spheres; off-screen meshes skipped on the CPU each frame. Toggle via
      `renderer.frustumCulling`, per-object opt-out via `object.frustumCulled`, and
      `renderer.culledCount` reports the last frame's skipped meshes.
- ✅ **GPU-driven culling** — opt-in `renderer.gpuCulling`: a compute shader
      (`@workgroup_size(64)`) reads per-slot world-space bounding spheres (updated
      each frame via `writeBuffer`) and six normalized frustum planes extracted from
      the view-projection matrix; it writes `instanceCount = 0` (culled) or `1`
      (visible) into a slot-indexed indirect draw buffer.  Opaque indexed meshes use
      `drawIndexedIndirect`; instanced/non-indexed meshes and OIT passes fall through
      to the CPU path.  Incompatible with `renderBundles` (silently uses the CPU path
      inside bundle encoders).  Sphere and indirect buffers grow × 2 when the model
      pool expands.
- ✅ **Render bundles** — opt-in `renderer.renderBundles` records the opaque draws into a
      `GPURenderBundle` and replays it, re-recording only when the draw set (or frame bind
      group) changes; per-object uniform buffers are still refreshed each frame so dynamic
      transforms stay correct.

---

## v0.4 — Visual quality ✅

A post-processing stack and richer materials.

- ✅ **Render graph / post pipeline** — opt-in `renderer.postProcessing`: the scene renders
      to an HDR (`rgba16float`) offscreen target (MSAA-resolved), then a chain of fullscreen
      triangle passes resolves to the swap chain. The material/line shaders skip their
      in-shader tonemap when a "linear output" frame flag is set.
- ✅ **Post effects** — ✅ a tone-mapping pass (ACES moved out of the material shader),
      ✅ **FXAA** (`renderer.fxaa`), and ✅ **bloom** (`renderer.bloom`: bright-pass +
      half-res separable-Gaussian blur, added before tonemap). ✅ **SSAO**
      (`renderer.ssao`, requires `postProcessing = true` + `sampleCount = 1`): an
      8-sample hemisphere SSAO pass reads the scene depth (sampled as `texture_depth_2d`),
      reconstructs view-space positions + depth-gradient normals via the inverse
      projection, rotates the kernel per pixel via a hash-based noise vector, projects
      samples with a soft range-check, blurs the occlusion result with the existing
      Gaussian passes, then multiplies it into the HDR output before tonemap; intensity
      tunable via `ssaoStrength`, `ssaoRadius`, `ssaoBias`. ✅ **TAA**
      (`renderer.taa`, requires `postProcessing = true` + `sampleCount = 1`): an
      8-sample Halton(2,3) sub-pixel jitter is baked into the uploaded projection
      matrix each frame; a resolve pass reconstructs world position from depth,
      reprojects it through the previous frame's unjittered view-projection
      (camera-motion reprojection — no per-object motion vectors), 3×3
      neighborhood-clamps the bilinear history sample against the current frame to
      limit ghosting, and blends history/current in HDR ping-pong targets before the
      tonemap chain; blend weight tunable via `taaBlend`, history auto-invalidated on
      resize or re-enable.
- ✅ **Material extensions** — ✅ **vertex colors** (glTF `COLOR_0`, VEC3/VEC4): an
      always-present per-vertex color stream (white default) multiplies base color in every
      vertex variant, so no new pipeline is needed. ✅ **`KHR_materials_clearcoat`** (factors):
      a second GGX specular lobe over the base, attenuating it by the coat's Fresnel; the
      `clearcoat`/`clearcoatRoughness` factors reuse spare material-uniform slots; ✅ **clearcoat
      textures** (`clearcoatMap` R-channel, `clearcoatRoughnessMap` G-channel) parsed from
      `KHR_materials_clearcoat.clearcoatTexture` / `clearcoatRoughnessTexture`, exported and
      sampled in the PBR shader (adds bindings 11–14 to the material bind group). ✅ **`KHR_materials_ior`** + **`KHR_materials_specular`** shape the dielectric
      F0 (IOR → F0, scaled/tinted by the specular factor & color); defaults (ior 1.5, white,
      1) reproduce the old flat 0.04. ✅ **`KHR_materials_sheen`** adds a Charlie-NDF
      retroreflective lobe for cloth (black sheen color = disabled). ✅ **`KHR_materials_transmission`**
      + **`_volume`** refract the environment/ambient through the surface with Beer–Lambert
      volume attenuation. ✅ **True screen-space refraction**: when `postProcessing=true`,
      the opaque HDR output is copied to a `sceneCapture` texture before the transparent
      pass; the refraction direction is projected to screen-space UV (worldPos + refr×thickness
      → clip → NDC → [0,1]²) and used to sample the snapshot, giving correct refractive
      distortion of opaque scene content. Falls back to env-map when post-processing is off.
- ✅ **Transparency (weighted-blended OIT)** — opt-in `renderer.oit` (requires the HDR post
      path): transparent meshes accumulate weighted premultiplied color into an `accum`
      target and product-of-(1−α) into a `reveal` target (depth-tested, no write), then a
      fullscreen pass composites them onto the HDR scene before tonemapping. A shared
      `shadeSurface()` feeds both the opaque `fs_main` and the OIT `fs_oit`. ✅ **MSAA
      support**: when `sampleCount > 1`, MSAA `accum`/`reveal` targets are created alongside
      the non-MSAA resolve targets; the OIT render pass uses them with `resolveTarget` so
      compositing always reads non-MSAA data; OIT pipeline keys include sampleCount.

---

## v0.5 — Assets & ecosystem ✅

Faster loads, broader inputs, and ergonomics.

- ✅ **Compressed geometry** — **`EXT_meshopt_compression`**: the loader decodes
      meshopt buffer views through a pluggable `MeshoptDecoder` (`loader.setMeshoptDecoder(MeshoptDecoder)`,
      the standard meshoptimizer module), then accessors read the decoded data. **`KHR_draco_mesh_compression`**:
      pluggable `DracoDecoder` interface (`loader.setDracoDecoder(decoder)`); the decoder
      receives the compressed buffer view bytes + Draco attribute ID map, returns indices +
      per-attribute typed arrays (Float32/Uint32); geometry is built from decoded data
      bypassing the stub accessors.
- ✅ **Compressed textures** — **`KHR_texture_basisu` / KTX2**: `KTX2Loader` parses the
      container, uploads uncompressed RGBA8 directly, and transcodes Basis (ETC1S/UASTC) via
      a pluggable transcoder. When `loader.setDevice(device)` is called and the transcoder
      implements `transcodeToCompressed`, the loader transcodes to BC7/ASTC 4×4/ETC2 (in
      that priority order, based on `device.features`) producing a `CompressedDataTexture`
      that `TextureManager` uploads directly without any RGBA8 intermediate.
- ✅ **Worker-based loading** — `WorkerGLTFLoader` (drop-in for `GLTFLoader`) spawns an
      inline Web Worker that fetches buffers and decodes images via `createImageBitmap` off
      the main thread, then transfers `ArrayBuffer`s and `ImageBitmap`s back as transferables;
      main thread assembles the scene graph. Decoder plugins (meshopt/Draco/KTX2) are forwarded
      to an underlying `GLTFLoader` and applied on the main thread after the worker returns.
      `GLTFLoader.buildFromPreloaded()` accepts pre-loaded resources from any source.
- ✅ **More cameras/controls** — `OrthographicCamera`; `FlyControls` (WASD + Q/E,
      mouse-look via drag or pointer lock, frame-rate-independent `update(delta)`).
- ✅ **Raycasting / picking** — `Ray` + `Raycaster` pick meshes; the broad phase
      rejects on the world bounding sphere, then survivors are tested per-triangle in
      local space (Möller–Trumbore), so hits carry an exact `point`, `faceIndex`, and a
      barycentric-interpolated `uv`. `setFromCamera` unprojects NDC; `precise = false`
      reverts to the coarse sphere test. Meshes at/above `bvhThreshold` triangles use a
      cached median-split **CPU `BVH`** that prunes subtrees the ray misses. ✅ **GPU
      id-buffer picking** (`renderer.pickAt(cssX, cssY, scene, camera)`): renders an
      offscreen `rgba8unorm` pass where each mesh is drawn with a flat RGB-encoded
      uint32 id, then copies the single clicked pixel to a read-back buffer; returns the
      hit `Mesh | null`. Id bind group uses 256-byte-aligned dynamic offsets so one
      `writeBuffer` call uploads all ids before the pass.
- ✅ **Helpers & debug** — `GridHelper`, `AxesHelper`, `Box3Helper`, and light gizmos
      (`DirectionalLightHelper`, `PointLightHelper`) via an unlit `LineSegments` +
      `LineBasicMaterial` path (line-list pipeline, per-vertex colors); plus a `Stats`
      FPS/ms overlay with a rolling graph (DOM-decoupled timing core).

---

## Backlog / under consideration ✅ (all resolved)

- ✅ **Clustered forward+** lighting (`renderer.clusteredLighting`) — a compute pass
  divides the view frustum into a 16×9×24 grid (screen tiles × logarithmic depth
  slices), computes each cluster's view-space AABB by scaling unprojected near-plane
  tile corners along their view rays, and bins light bounding spheres into per-cluster
  lists (count + up to 32 light indices, binding 14). Fragments derive their cluster
  from the pixel position and view-space depth and shade only that cluster's lights;
  `MAX_LIGHTS` rises to 256. Directional and infinite-range lights land in every
  cluster; perspective cameras only. Disabled, the shader falls back to the
  loop-over-all-lights path with a dummy buffer bound.
- ✅ **glTF export** (`GLTFExporter`) — writes node hierarchy + TRS, mesh geometry
  (position/normal/uv/color/indices), `StandardMaterial` PBR factors + **textures**
  (map/normalMap/metalnessRoughnessMap/emissiveMap/occlusionMap encoded as PNG into the
  binary buffer with samplers), clearcoat/ior/specular/sheen extensions, **morph targets**
  (deltas + default weights + `targetNames`), keyframe **animation** clips
  (translation/rotation/scale/weights, all interpolations), and **skinning** (JOINTS/WEIGHTS,
  skins + inverse bind matrices) to a binary `.glb`. All export methods are async to support
  texture encoding via `OffscreenCanvas`.
- ✅ **Lightweight scene format** (`SceneSerializer`) — lossless JSON round-trip of the scene
  graph (`Object3D`/`Mesh`/`LineSegments`/lights, transforms, `BufferGeometry` attributes,
  `StandardMaterial`/`LineBasicMaterial`), with geometry/material de-duplication preserved
  across load. Verified by a serialize→stringify→parse→deserialize round-trip.
- ✅ **`ShaderMaterial`** (chosen over a node-based material graph — vela is AI-first,
  and AI agents write WGSL directly; no visual editor is planned). The user supplies
  one WGSL function `fn surface(in : VSOut) -> Surface` (albedo/alpha/metalness/
  roughness/emissive/normal/occlusion, seeded by `defaultSurface(in)`); the engine
  wraps it with the shared vertex variants and the full PBR lighting path (all light
  kinds, directional/spot/point shadows, IBL, clustered lighting, OIT, tonemap flags)
  extracted into a reusable `SHADE_HELPERS` WGSL chunk. Custom uniforms are a plain
  JS object auto-packed per WGSL layout rules into one group-2 uniform buffer
  (`u.<name>`), re-uploaded every frame with no dirty flags; shape changes recompile
  automatically. `elapsedTime()` is built in. WGSL compile errors print with the
  offending generated-source line. See docs/SHADER_MATERIAL.md. Also fixed: material/
  line pipelines now key on the actual scene color format, so they are valid against
  the HDR (`rgba16float`) target when `postProcessing` is enabled.

---

## v0.6 — Game-visual table stakes ⬜

What a game needs from its renderer beyond a lit model viewer. Everything stays
declarative: an object literal to set up, no builder chains, no hidden update flags.

- 🚧 **Skybox & procedural sky** — ✅ **skybox** (`scene.skybox = true`): a
      fullscreen-triangle pass at `depth = 1` (depthCompare `less-equal`, no depth
      write) drawn after the opaques and before transparents; the vertex stage
      reconstructs world rays from the projection diagonal + transposed view rotation
      and the fragment samples the raw equirect environment (its own bind group —
      frame binding 4 holds the low-res prefiltered map when IBL is active).
      `scene.backgroundBlur` (0..1, via `clusterParams.w`) picks a mip. ⬜ Procedural
      sky (`scene.sky = { sunDirection, turbidity, ... }`): a small analytic sky
      (Preetham/Hosek-style fit) generated into an equirect texture and fed into the
      IBL prefilter so lighting matches the visuals.
- ⬜ **Fog** — `scene.fog = { color, near, far }` (linear) or `{ color, density }`
      (exponential), plus optional height falloff. Implemented in `SHADE_HELPERS` as a
      final `applyFog(color, worldPos)` mix using view distance, so StandardMaterial,
      ShaderMaterial, and the line path all pick it up; frame uniform gains a fog vec4
      pair (spare `clusterParams.w` + a new slot).
- ⬜ **GPU particles** — `ParticleSystem` with a declarative emitter config (rate,
      lifetime, velocity/spread, gravity, size/color-over-life as gradient stops, world
      or local space). A compute pass integrates a fixed-capacity particle pool in a
      storage buffer (dead particles recycled via an atomic freelist); rendering is one
      instanced quad draw reading the pool, depth-tested but unsorted (additive) or
      OIT-composited (alpha). No per-particle JS objects — the hot path never allocates.
- ⬜ **Sprites & SDF text** — billboarded quads (`Sprite`) batched into one instanced
      draw per texture; `TextMesh` renders strings via a runtime-generated SDF atlas
      (`OffscreenCanvas` rasterization → distance transform, cached per font), one quad
      per glyph in the same batcher. Both work in world space and as screen-space
      overlays (HUD: `sprite.screenSpace = true`, sized in CSS pixels).
- ⬜ **Render-to-texture** — `const rt = new RenderTarget(w, h)` +
      `renderer.render(scene, camera, rt)`: renders the full pipeline (including post)
      into an offscreen color texture usable as any material map (mirrors, portals,
      minimaps, security cams). Depth/MSAA resources sized per target and cached;
      `rt.texture` plugs into `StandardMaterial.map` or a future ShaderMaterial texture
      uniform.

---

## v0.7 — The agent feedback loop ⬜

vela's developers are AI agents: they cannot see the canvas, and they iterate through
text. This tier makes rendering results readable, diffable, and explainable without a
human looking at pixels — the core AI-first differentiator.

- ⬜ **Pixel readback & screenshots** — `renderer.readPixels(x?, y?, w?, h?)` →
      `Uint8ClampedArray` and `renderer.screenshot()` → PNG blob/data-URL, implemented
      with a `COPY_SRC` hook on the final target + a mapped read-back buffer (reuses the
      id-picking machinery). Works headless via `OffscreenCanvas`.
- ⬜ **Deterministic mode** — `renderer.deterministic = true`: TAA jitter sequence
      restarts from a fixed seed, `elapsedTime()` is driven by a settable
      `renderer.time` instead of the wall clock, and any future stochastic effects key
      off one seed — so the same scene always produces the same pixels.
- ⬜ **Golden-image testing** — `vela/test` helper: `await expectFrame(renderer, scene,
      camera, 'golden/lava.png', { tolerance })` renders deterministically, compares
      with a perceptual diff (per-channel + SSIM-lite), writes the actual + diff images
      on failure. Makes "verify the change visually" a unit test an agent can run.
- ⬜ **`renderer.diagnose(scene, camera)`** — structured triage of the classic
      black-screen causes, returned as `{ severity, code, message, fix }[]`: no
      lights/zero intensity, camera frustum missing the scene bounds (with the actual
      distance and a suggested position), NaN/zero-scale transforms, model scale outliers
      (>1000× scene median), `transparent` without alpha < 1, `oit`/`ssao`/`taa` flags
      whose prerequisites (postProcessing, sampleCount) aren't met, materials referencing
      destroyed textures. Every message names the offending object and the one-line fix.
- ⬜ **`scene.describe()` / `renderer.report()`** — JSON introspection an LLM can
      reason over: node hierarchy summary (counts by type, named nodes), world bounds,
      material/light/texture inventories, last-frame stats (draw calls, triangles,
      culled counts, GPU buffer/texture memory estimates), and per-mesh visibility
      (in-frustum? culled by what?).
- ⬜ **`llms.txt` + generated API reference** — a single context-window-friendly page
      of the whole public API (signatures + one-line docs, generated from the `.d.ts`
      surface by a script in `scripts/`), published at the repo root and `docs/llms.txt`
      per the llms.txt convention.

---

## v0.8 — AI-extensible pipeline ⬜

Extend the ShaderMaterial pattern — "write one WGSL function, the engine does the
rest" — to the remaining programmable surfaces.

- ⬜ **ShaderMaterial v2** — texture uniforms (`uniforms: { noise: texture }` →
      auto-bound `t_noise`/`s_noise` pair on group 2, layout regenerated on shape change
      like scalar uniforms) and an optional vertex hook (`fn displace(pos : vec3<f32>,
      in : VSIn) -> vec3<f32>`) spliced into all four vertex variants for displacement
      and wind effects.
- ⬜ **ShaderPass** — custom fullscreen post effects: `renderer.passes.push(new
      ShaderPass({ fragment, uniforms }))` with the same auto-packed uniform object;
      the pass receives the previous stage's HDR view + depth and is inserted into the
      PostProcessing chain before tonemap. Compile errors report exactly like
      ShaderMaterial.
- ⬜ **Compute API** — `new ComputeTask({ code, workgroups, buffers: { particles:
      storage(...), params: uniform({...}) } })` with declarative buffer/uniform
      bindings (auto layout, same packing rules), `task.dispatch(encoder?)`, and
      `task.read('buffer')` for results — GPU sims (boids, erosion, procedural
      geometry) without touching raw WebGPU.

---

## v0.9 — Scale & robustness ⬜

- ⬜ **LOD** — `LOD` node with distance-banded children (hysteresis to avoid popping);
      selection happens in `collect()` against the camera distance already computed for
      transparency sorting.
- ⬜ **GPU occlusion culling** — two-phase: render last frame's visible set, build a
      hi-Z mip pyramid from depth, then test the remaining bounds against it in the
      existing GPU-cull compute pass (extends the indirect-draw path).
- ⬜ **Performance advisor** — `renderer.report()` grows actionable suggestions with
      the same `{ code, message, fix }` shape as `diagnose()`: "412 draws share one
      geometry+material — use InstancedMesh", "shadow map re-rendered but no caster
      moved", "postProcessing off but bloom set". An agent has no perf intuition; the
      engine lends it some.
- ⬜ **Lifecycle hardening** — device-lost recovery (re-init device, rebuild all cached
      GPU resources from their CPU-side sources, which the cache design already keeps),
      a `dispose()` audit across geometry/texture/material/render-target paths, and a
      debug allocation tracker (`renderer.resources()`) so long-running agent sessions
      don't leak.

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
6. **Every feature must be usable blind.** vela's developers are AI agents that cannot
   see the canvas: each feature is declarative to set up, introspectable as data
   (`describe()`/`report()`), and verifiable without a human looking at pixels
   (deterministic rendering + readback). Error messages name the offending object and
   the one-line fix.
