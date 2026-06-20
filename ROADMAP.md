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

## v0.6 — Game-visual table stakes ✅

What a game needs from its renderer beyond a lit model viewer. Everything stays
declarative: an object literal to set up, no builder chains, no hidden update flags.

- ✅ **Skybox & procedural sky** — ✅ **skybox** (`scene.skybox = true`): a
      fullscreen-triangle pass at `depth = 1` (depthCompare `less-equal`, no depth
      write) drawn after the opaques and before transparents; the vertex stage
      reconstructs world rays from the projection diagonal + transposed view rotation
      and the fragment samples the raw equirect environment (its own bind group —
      frame binding 4 holds the low-res prefiltered map when IBL is active).
      `scene.backgroundBlur` (0..1, via `clusterParams.w`) picks a mip. ✅ **Procedural
      sky** (`scene.sky = { sunDirection, turbidity? }`): a compute pass evaluates the
      Preetham analytic daylight model (Perez distributions for Y/x/y, zenith fits,
      xyY→linear sRGB, soft sun disc, ground fade) into a 256×128 equirect
      `rgba16float` storage texture, which routes through the standard env path — IBL
      prefilter + skybox — so lighting matches the visuals. Param changes are detected
      by value each frame (no flags); mutating `sunDirection` animates day/night and
      re-convolves IBL automatically. `scene.environment` takes precedence when both
      are set.
- ✅ **Fog** — `scene.fog = { color, near, far }` (linear) or `{ color, density }`
      (exponential-squared), plus optional `heightFalloff` (fog thins with altitude).
      Implemented in `SHADE_HELPERS` as a final `applyFog(color, worldPos)` mix on
      view distance, so StandardMaterial and ShaderMaterial both pick it up; the frame
      uniform grew to 320 bytes with `fogColor` (rgb + mode) and `fogParams` vec4s.
      Debug helpers (the line path) deliberately stay unfogged so grids/gizmos remain
      readable. Also fixed the line shader's linear-out check to test bit 0 of the
      `envParams.w` bit field instead of `>= 0.5` (it misfired whenever IBL was active).
- ✅ **GPU particles** — `ParticleSystem extends Object3D` with a declarative emitter
      config (rate, lifetime, velocity/spread, gravity, size/color/opacity over life as
      `[start, end]` pairs, additive or alpha blending), re-read every frame with no
      flags; emission happens at the object's world position. A compute pass
      (`@workgroup_size(64)`, one invocation per pool slot) integrates a fixed-capacity
      48-byte-stride pool in a storage buffer; emission uses a ring cursor — slots in
      the wrapped window respawn at the emitter (simpler than an atomic freelist, same
      capability: oldest particles recycle when full; a zeroed buffer means all-dead).
      Rendering is one 6-vertex instanced draw per system: camera-facing soft-disc
      quads built from the view-matrix basis, depth-tested but not written,
      premultiplied additive/alpha blending, tonemapped like the material path. The
      per-frame upload path reuses scratch arrays — no per-particle JS objects, no
      hot-loop allocation.
- ✅ **Sprites & SDF text** — ✅ **sprites**: `Sprite extends Object3D` (texture,
      color/opacity tint, `size`/`offset`, `screenSpace`) batched per (texture, mode)
      into one 6-vertex instanced draw from a 64-byte-stride instance storage buffer
      (grown ×2, rebuilt per frame with no per-sprite GPU objects). World-space
      sprites billboard from the view-matrix basis and depth-test against the scene;
      `screenSpace = true` projects the anchor then offsets in CSS pixels
      (pixel-ratio-scaled CPU-side) with the depth test off, drawn after everything as
      a HUD overlay. Premultiplied alpha, literal color like the helper path.
      ✅ **SDF text** (`TextMesh`): glyphs rasterize lazily via Canvas2D into a
      per-font 1024² atlas (64px cells, 256 glyphs) with an exact Felzenszwalb
      Euclidean distance transform (offline-verified against analytic fields);
      distance lives in alpha and the texture version bumps per new glyph so the
      atlas re-uploads automatically. Each character becomes one instance in the
      shared sprite batcher (per-glyph uv rect + pen/baseline offsets, multi-line
      `\n`, left/center/right anchors); the fragment thresholds the SDF at 0.5
      with `fwidth` antialiasing, so text stays crisp at any scale in both world
      and screen space.
- ✅ **Render-to-texture** — `const rt = new RenderTarget(w, h)` +
      `renderer.render(scene, camera, rt)`: renders the scene pipeline (lights,
      shadows, sky, fog, particles, sprites/text, clustered lighting) into an
      offscreen `rgba8unorm` color texture with per-target cached depth/MSAA
      resources; `rt.texture` plugs into any material map (mirrors, portals,
      minimaps, security cams) via a TextureManager external-entry registration and
      an `-srgb` sampling view, so the tonemapped sRGB bytes decode back to linear.
      RT passes use the direct pipeline with in-shader tonemapping — the post chain
      (and TAA history/jitter state, which RT renders deliberately don't touch)
      belongs to the canvas target. Cluster tile dims and shader `elapsedTime()`
      follow the active target's size.

---

## v0.7 — The agent feedback loop ✅

vela's developers are AI agents: they cannot see the canvas, and they iterate through
text. This tier makes rendering results readable, diffable, and explainable without a
human looking at pixels — the core AI-first differentiator.

- ✅ **Pixel readback & screenshots** — `await renderer.readPixels()` resolves with
      the next presented canvas frame (full post chain included: the swap chain is
      configured with `COPY_SRC` and pending captures copy into a mapped read-back
      buffer just before present, with row-padding strip + BGRA→RGBA swizzle);
      `readPixels(rt)` reads a RenderTarget immediately. `renderer.screenshot()`
      wraps either in a PNG Blob via `OffscreenCanvas.convertToBlob`. Errors name
      the fix (e.g. reading a never-rendered target).
- ✅ **Deterministic mode** — `renderer.deterministic = true`: shader
      `elapsedTime()`, particle integration dt, and particle PCG seeds all derive
      from a settable `renderer.time` (advance it manually, e.g. `+= 1/60`) instead
      of the wall clock, and the TAA jitter sequence + history restart from a fixed
      point on enable — so the same scene + time always produces the same pixels.
- ✅ **Golden-image testing** — `await expectFrame(renderer, scene, camera,
      'golden/lava.png', { frames, channelTolerance, maxDiffRatio })` renders
      deterministically (warmup frames + fixed time), captures via `readPixels()`,
      and compares per-channel with a differing-pixel ratio budget; on mismatch it
      throws `FrameMismatchError` carrying metrics plus the actual and red-highlighted
      diff frames as PNG blobs (object URLs logged for instant viewing, and the
      message says how to bless a new golden). `comparePixels` is pure and
      offline-verified (identical/1-px/noise/size-mismatch cases); `captureFrame`,
      `loadPixels`, `pixelsToPNG` are exported separately for custom harnesses.
- ✅ **`renderer.diagnose(scene, camera)`** — structured triage of the classic
      black-screen causes, returned as `{ severity, code, message, fix }[]`: empty
      scene, no lights/ambient/env, zero-intensity or black lights, camera frustum
      missing the scene bounds (with the actual distance and a copy-pasteable
      suggested position), scene-beyond-`camera.far` (distinguished from bad aim via
      a facing test), invalid/degenerate near/far + depth-precision warnings, aspect
      mismatch with the canvas, NaN world matrices, zero scale components, scale
      outliers (>1000× the scene median radius), missing/unsupported materials
      (which the renderer would silently skip), `transparent` without an alpha
      source, `oit`/`ssao`/`taa` prerequisite violations, skybox without an
      environment, shadow-casting lights with `renderer.shadows` off, and inverted
      fog ranges. Pure scene-graph math in `diagnoseScene()` (no GPU) —
      offline-tested against nine synthetic scenes including the healthy-scene
      no-findings case.
- ✅ **`describeScene()` / `renderer.report()`** — JSON introspection an LLM can
      reason over. `describeScene(scene, camera?)` (pure, offline-tested): counts by
      node/light type, world bounds (center/size/radius), de-duplicated material
      inventory with user counts, environment/skybox/fog state, named nodes with
      world positions, and — given a camera — per-named-mesh `inFrustum` plus
      aggregate visibility counts. `renderer.report()` (last frame): draw calls,
      triangles (instancing-aware), opaque/transparent/culled mesh counts, light
      count, particle pool capacity, sprite batches/instances, shadow re-draw cost,
      estimated GPU texture memory (tracked by the TextureManager incl. mip
      overhead), and active feature flags.
- ✅ **`llms.txt` + generated API reference** — `npm run docs:llms`
      (`scripts/generate-llms.mjs`) extracts the entire public API from
      `src/index.ts` via the TypeScript checker — classes with own members +
      one-line docs (inherited members listed once on the base), function
      signatures, and interfaces/type aliases verbatim — and emits a single
      ~50 KB context-window-friendly page (quick-start example + the
      stuck-agent toolkit pointers) at the repo root and `docs/llms.txt` per
      the llms.txt convention.

---

## v0.8 — AI-extensible pipeline ✅

Extend the ShaderMaterial pattern — "write one WGSL function, the engine does the
rest" — to the remaining programmable surfaces.

- ✅ **ShaderMaterial v2** — **texture uniforms**: a `Texture` value in `uniforms`
      becomes a `t_<name>`/`s_<name>` texture+sampler pair on group 2 (bindings
      1,2 / 3,4 / …, scalar buffer at 0 only when present). The group-2 bind layout
      and pipeline layout are built dynamically per `(hasBuffer, textureCount)` shape;
      the bind group rebuilds when a texture's id/version changes, the shader
      recompiles when the uniform shape changes. **Vertex hook**: an optional
      `fn displace(position, in : VSIn) -> vec3<f32>` is spliced into the static and
      instanced vertex stages before the model transform (skinned/morph keep their
      built-in stages). Offline-verified: all four variants + texture-only +
      empty-uniform cases parse via wgsl_reflect with correct group-2 bindings.
- ✅ **ShaderPass** — custom fullscreen post effects: `renderer.passes.push(new
      ShaderPass({ effect, uniforms }))`. Each enabled pass runs in order in HDR
      linear space after bloom/SSAO/TAA and before tonemap, ping-ponging two
      full-res HDR buffers. The user writes `fn effect(uv) -> vec4<f32>` with
      `sceneColor(uv)`/`sceneDepth(uv)` helpers, `pp.resolution`/`pp.time`, and the
      same auto-packed `u.<name>` / `t_<name>` uniforms as ShaderMaterial (in group
      1; `computeUniformLayout` gained a group parameter). Per-pass pipeline + bind
      group are cached and rebuilt on version/shape/texture change; compile errors
      print the offending generated line like ShaderMaterial. Read↔write aliasing is
      avoided by flipping the target. Offline-verified: no-uniform, scalar+texture,
      and group-binding cases parse via wgsl_reflect.
- ✅ **Compute API** — `new ComputeTask({ code, entryPoint?, workgroups,
      bindings: { data: storage({ size, data?, readable? }), params: uniform({...}) } })`.
      Named bindings map to `@group(0)` in declaration order (binding 0,1,2,…); the
      bind group layout is built automatically and uniform values are auto-packed via
      the shared `computeUniformLayout`/`packUniforms`. `await task.init(device)`
      compiles (with the same line-numbered error reporting as ShaderMaterial/
      ShaderPass); `task.dispatch(encoder?)` folds into a frame or runs standalone,
      `task.run()` is the standalone shortcut, `task.write()`/`updateUniform()` push
      new data, and `await task.read(name)` copies a `readable` storage buffer back to
      an `ArrayBuffer`. `task.buffer(name)` exposes the raw `GPUBuffer` for
      interop. No raw WebGPU plumbing for boids/sims/procedural-geometry jobs.

---

## v0.9 — Scale & robustness ✅

- ✅ **LOD** — `LOD extends Object3D` with `addLevel(object, distance, hysteresis?)`
      (levels auto-sorted by distance). Before `collect()`, the renderer walks the
      scene and calls `lod.update(cameraWorldPos)`, which toggles child `.visible` so
      the active level flows through frustum culling and every other traversal
      unchanged. Cold-start picks the natural level; thereafter hysteresis is a
      bidirectional deadband — switch up only past `distance + hysteresis`, down only
      below `distance - hysteresis` — eliminating boundary flicker. `autoUpdate = false`
      hands control to manual `update()` calls. Selection + hysteresis offline-verified.
- ✅ **GPU occlusion culling** (opt-in `renderer.occlusionCulling`, builds on
      `gpuCulling`, needs `sampleCount = 1`) — single-phase previous-frame hi-Z. Each
      frame, after the scene pass, a max-depth pyramid is built from the depth buffer
      (mip 0 = a depth copy, each coarser mip = a 2×2 max-reduce fullscreen pass into
      an `r32float` mip chain). Next frame the cull compute shader projects each
      bounding sphere through the pyramid's matching view-projection, picks the mip
      whose texel covers the sphere's screen AABB, and culls it only if its nearest
      depth is behind the max occluder depth (conservative). The cull bind group gains
      the hi-Z texture+sampler (a 1×1 dummy when inactive); a flag in `CullParams`
      keeps the frustum-only path identical when occlusion is off. **Caveat:** uses the
      prior frame's depth, so geometry revealed by fast camera motion can pop one frame
      late; verified offline (WGSL parse + struct offsets) but not yet on GPU hardware.
- ✅ **Performance advisor** — `renderer.report().suggestions` is a list of
      `{ code, message, fix }` items (same shape as `diagnose()`) computed from the
      last frame: an instancing opportunity (≥16 meshes sharing one geometry+material
      drawn separately → InstancedMesh), high draw-call count without render bundles,
      many ranged lights without clustered lighting, a post effect enabled while
      `postProcessing` is off, and MSAA blocking SSAO/TAA. An agent has no perf
      intuition; the engine lends it some.
- ✅ **Lifecycle hardening** — all device-owned GPU resource creation + cache resets
      are centralized in one private `setupDeviceState()` (the single source of truth
      for "what lives on the GPU"), shared by `init()` and `restoreContext()`. The
      device-lost handler distinguishes intentional teardown (`reason === 'destroyed'`)
      from real loss, pauses rendering (`render()` early-returns while lost), fires an
      `onDeviceLost` hook, and prints how to recover. `await renderer.restoreContext()`
      requests a fresh device and rebuilds everything — geometry/material/texture caches
      repopulate lazily from their surviving CPU-side data. `dispose()` destroys the
      device (freeing all its resources) and blocks further rendering; `isDisposed`/
      `isDeviceLost` expose state and `resources()` reports tracked GPU usage (texture
      bytes, sprite batches, font atlases, model-pool slots) so long-running agent sessions
      don't leak.

---

## v0.10 — Generic hooks that unblock NPR ⬜

Cel/comic/anime looks are **applications**, not engine features (principle #7): they live
in app code on `ShaderMaterial` / `ShaderPass` / `ComputeTask`, as the gltf-viewer's
`comic.*` already demonstrates. This tier is the inverse of the now-removed `celShading`
flag — it closes the *general* API gaps those apps hit, with capabilities any renderer
client could use, never a built-in style.

- ✅ **Lighting terms inside `ShaderMaterial`** — two optional WGSL hooks expose the
      engine's lighting to a custom shade function so toon ramps, wrap/half-Lambert, and
      posterized highlights remap it instead of re-deriving it. `light(s, l : LightSample)`
      is called once per light that reaches the fragment — the engine still resolves the
      clustered list, distance/cone attenuation, and shadow visibility and hands them over
      as `l.radiance`, with the **raw** (signed) `l.NoL` plus `L/V/H/NoV/NoH/VoH` so wrap
      lighting works; the engine sums the returned per-light contributions.
      `ambient(s, ind : IndirectSample)` replaces the indirect (IBL/flat ambient) term.
      `defaultLight(s, l)` / `defaultIndirect(s, ind)` reproduce the engine's PBR result for
      blending. Generic — no ramp or style baked into the engine. Offline-verified (all four
      vertex variants × {none, light, ambient, both} parse with the right entry points).
- ✅ **Normal / linear-depth targets for `ShaderPass`** — a pass declares
      `inputs: ['normal', 'linearDepth']` to opt into one packed RGBA16F scene target
      (world-space shading normal in RGB, normalized linear depth in A). Helpers expose
      `sceneWorldNormal()`, `sceneViewNormal()`, and positive world-unit
      `sceneLinearDepth()`; background depth is the camera far plane. The renderer adds
      the MRT + pipeline bit only while an enabled pass requests it, with one lazy cached
      target (including MSAA resolve), so the default single-target path is unchanged.
      StandardMaterial normal maps and ShaderMaterial custom normals both feed the target.
      Offline-verified across all four vertex variants and both single/MRT entry points.
- ✅ **Shell / inverted-hull draw** — `mesh.shell = { material, thickness }` re-draws the
      geometry after the opaque pass with back faces (front-culled) extruded `thickness`
      world units along the vertex normal. The extrusion is folded into the shared vertex
      stages as a per-object `params.x` (0 for normal draws), so it works for static,
      skinned, and morph meshes — not just the ones `ShaderMaterial.displace` reaches — and
      adds no extra pipeline variant (the only render-state change is a forced `front` cull,
      threaded through `PipelineCache.get`/`getCustom` as a cull override). The shell uses
      its own model-pool slot (the main draw needs thickness 0 in the same frame) and reuses
      the standard PBR/Surface fragment, so the *look* lives entirely in `material` (flat
      emissive = outline, fresnel = rim, texture = fur); the engine only provides the
      extruded back-face draw. Not supported on `InstancedMesh`. Offline-verified (Model
      struct 144 B with thickness at offset 128; all PBR/shadow/id/Surface shaders parse).
- ✅ **1-D ramp / LUT texture ergonomics** — `gradientTexture(stops, width?)` builds a
      `width × 1` linear `DataTexture` (clamp-wrapped, linearly filtered) from `ColorInput`
      stops (+ optional per-stop opacity) — the generic primitive behind toon ramps,
      gradient maps, and color LUTs. Drops straight into a `ShaderMaterial`/`ShaderPass`
      texture uniform, sampled at `(t, 0.5)`. No hidden color space (stops are linear).
      Offline-verified (endpoints, midpoint, multi-stop, opacity, clamp, unsorted input).

## v0.11 — Advanced lighting & global illumination ⬜

The lighting model tops out at punctual lights + a single environment. Real scenes want
local reflections and bounced light. Each item leans on existing GPU machinery (clustered
binning, the IBL prefilter compute path, RenderTarget cube capture).

- ✅ **Screen-space reflections (SSR)** — `renderer.ssr` adds a 48-step, quadratic-spaced
      view-space ray march in HDR before TAA/custom passes/tonemap. It consumes the packed
      world-normal target and the current frame's max-depth hi-Z pyramid (shared with
      occlusion culling), chooses coarse mips from each step's screen footprint, then
      binary-refines candidate hits at mip 0. Hits sample the HDR scene with edge/distance/
      Schlick-Fresnel fading; miss/off-screen rays leave the original PBR result untouched,
      preserving its IBL specular fallback. `ssrIntensity`, `ssrMaxDistance`, and
      `ssrThickness` tune it. Requires `postProcessing = true`, `sampleCount = 1`; both
      prerequisites are covered by `diagnose()`/performance suggestions. Offline-verified
      (WGSL parse, six bindings, 224-byte params); TAA consumes the SSR result to stabilize
      ray-march shimmer.
- ✅ **Reflection probes** — `ReflectionProbe` captures the scene through six RenderTarget
      passes (+X/−X/+Y/−Y/+Z/−Z), converts them to an HDR equirectangular map, and reuses
      `IBLPrefilter` for six-level GGX specular prefiltering. The first four visible probes
      occupy one texture array (staying within baseline sampled-texture limits); fragments
      select and distance-blend their nearest two inside each probe radius, falling back to
      the global environment outside. `refresh: 'static'` captures once (`needsUpdate` bakes
      again); `'every-n-frames'` uses `refreshInterval`. `describeScene()` reports probe count.
- ✅ **Diffuse GI via irradiance probes** — `IrradianceProbeGrid` defines a world-axis-
      aligned regular grid; `await renderer.bakeIrradianceProbes(grid, scene)` captures six
      faces per point and projects them on the GPU into nine cosine-convolved RGB SH-L2
      coefficients. StandardMaterial and ShaderMaterial trilinearly sample the first baked
      visible grid, replacing global diffuse ambient/IBL inside its bounds while leaving
      specular reflections unchanged. Coefficients read back to CPU for deterministic
      offline baking and round-trip through `SceneSerializer`; `diagnose()` identifies an
      unbaked grid and `describeScene()` reports grid count.
- ✅ **Cascaded shadow maps (CSM)** — `renderer.shadowCascades` (1–4, default 1)
      applies a practical logarithmic/uniform split (`shadowCascadeLambda`) to perspective
      camera frusta. Each cascade gets a fitted light matrix and a tile in a 2×2 directional
      depth atlas; fragments select by view depth and cross-fade over
      `shadowCascadeBlend` of the split range. One cascade preserves the original full-map
      path, non-perspective cameras fall back with a `diagnose()` warning, and
      `renderer.report().shadowDraws` includes every cascade re-draw. Offline-verified:
      uniform/log split invariants, 288-byte cascade block, and all material variants parse.
- ✅ **Volumetric fog & light shafts** — `renderer.volumetricFog` ray-marches the existing
      linear/exp²/height fog through a 16×9×24 `rgba16float` froxel volume before the scene
      pass. Each froxel reuses its logarithmic clustered light list (or all lights when
      clustering is off); punctual attenuation and spot cones contribute in-scattering,
      while the directional CSM atlas occludes the march to form light shafts. The volume
      stores accumulated scattering + transmittance and `applyFog()` samples it by screen
      position/linear depth. Disabled, the prior analytic path is unchanged. `diagnose()`
      identifies a missing `scene.fog`; all PBR/ShaderMaterial variants and the compute
      shader parse offline.

## v0.12 — Cinematic camera & color ⬜

Camera-domain effects and final-image control — the layer between a correct render and a
shot. All slot into the existing HDR post chain before/after tonemap.

- ✅ **Motion vectors → motion blur** — when TAA, motion blur, or scene-input passes are
      active, the scene MRT includes an `rg16float` current→previous UV velocity target.
      The 208-byte model slot retains the previous model matrix; instanced matrices,
      skeleton bone matrices, and morph weights each retain a previous-frame GPU buffer,
      while the 384-byte frame block carries the previous camera view-projection. TAA now
      reprojects history with this per-object velocity. `renderer.motionBlur` runs a 3×3
      velocity-neighborhood max followed by a nine-tap HDR reconstruction blur
      (`motionBlurStrength`), before TAA/custom passes/tonemap. Requires post-processing +
      sampleCount 1, covered by `diagnose()`. Offline-verified across static/skinned/
      instanced/morph PBR and ShaderMaterial variants plus TAA/blur shaders.
- ✅ **Depth of field** — `renderer.depthOfField` derives a circle of confusion from the
      scene depth plus `focusDistance`/`aperture`, bounded by `maxBlur` pixels. A 16-sample
      golden-angle HDR bokeh reconstruction handles near and far blur with a depth-aware
      foreground guard, before motion blur/TAA/custom passes/tonemap. Requires
      post-processing + sampleCount 1 and is covered by `diagnose()`; shader interface and
      production build verified offline.
- 🚧 **Color grading & display transforms** — ✅ `renderer.toneMapping` selects the output
      transform: `'aces'` filmic, `'agx'` (AgX operator — minimal log-sigmoid approximation,
      gentler highlight desaturation; bloom-paired entries `fs_tonemapAgx[Bloom]`), or
      `'none'` linear. ✅ **3-D LUT** (`renderer.colorLUT = ColorLUT.parseCube(text)`): a
      `.cube` LUT uploaded as an `rgba16float` 3-D texture and applied as the final post
      pass (after tonemap/FXAA, in display space — hardware-trilinear, blend by `strength`);
      the no-LUT path stays byte-identical. ⬜ Lift/gamma/gain + saturation as cheap analytic
      ops. Keeps the "linear output" flag contract so material shaders stay tonemap-agnostic.
- ⬜ **Lens & exposure** — **auto-exposure** (a compute histogram of the HDR target →
      adapted EV, smoothed over time; deterministic mode pins it), plus vignette,
      chromatic aberration, and an optional lens-flare/bloom-streak — all small
      `ShaderPass`-shaped effects gated on `postProcessing`.

## v0.13 — Animation systems ✅

Playback exists (`AnimationMixer`); authoring runtime motion does not. This tier makes the
mixer a real animation system, staying data-first so an agent can wire state without a UI.

- ✅ **Blending & cross-fade** — the mixer is action-based: each clip runs as an
      `AnimationAction` (own time/weight/loop/mask). `update()` samples every enabled action
      and blends per (node, path) — weighted average for translation/scale/morph weights,
      hemisphere-aligned normalized weighted sum (nlerp) for rotation. `mixer.add(clip, w)`,
      `mixer.crossFadeTo(clip, dur)` (fades the new action 0→1 while current actions fade out
      and prune), and `action.setMask(targets)` per-bone masks (e.g. an upper-body action
      over a full-body walk). `KeyframeTrack.evaluate(time, out)` was split out from `sample`
      so tracks feed the blend without touching the node; single-clip `play(clip)` is
      unchanged. Bindings are cached, so a steady action set allocates nothing per frame.
      Offline-verified (average / weighted / nlerp / crossfade-prune / mask).
- ✅ **Additive layers** — `action.additive = true` (+ `referenceTime`) adds the clip's
      delta from its reference pose on top of the blended base rather than averaging in:
      translation/scale/weights add `weight·(cur − ref)`, rotation applies a weight-scaled
      delta quaternion (`base · slerp(id, cur·ref⁻¹, weight)`). Evaluated in a second pass
      after the base blend, so a recoil/breathing layer rides on a walk (drives the same
      nodes for no drift). Offline-verified (translation, rotation, weight scaling, no drift).
- ✅ **Blend trees / state machine** — `AnimationStateMachine` takes a declarative
      `{ states, transitions, parameters }` graph: states are single clips or **1-D blend
      spaces** (nearest-two weighting by a `speed`-style parameter), transitions fire on
      parameter `Condition`s (`from: '*'` = any) and crossfade. Each frame it accumulates
      per-clip weights into its `AnimationMixer` (reusing all the blend/crossfade math) and
      `describe()` returns a JSON snapshot (current state, parameters, active transition +
      progress, active clip weights). Offline-verified (blend interpolation, clamping,
      crossfade, transitions). 2-D blend spaces and phase-synced blending are follow-ups.
- ✅ **Two-bone IK** — `solveTwoBoneIK(root, mid, end, target, { pole })` rotates the two
      joint local quaternions so `end` reaches a world-space `target`, bending toward an
      optional `pole`. Geometric solve: place the mid joint on the law-of-cosines circle in
      the aim×pole plane, then orient each bone with `setFromUnitVectors` (robust to a fully
      straight starting pose; out-of-reach extends straight). Apply after the mixer writes
      the pose. Offline-verified (reach across targets, out-of-reach, pole direction from a
      straight limb, re-solve stability).
- ✅ **Animation events** — `clip.addEvent(time, name)` adds sorted timeline markers;
      `action.onEvent = (name, time) => …` fires as the mixer crosses them, loop-aware
      (a wrap from near the end past 0 still fires early events) and direction-aware.
      Gameplay (footsteps, hit-frames) hangs off the timeline without polling.
      Offline-verified (ordering, crossing, no-double-fire, loop-wrap).

## v0.14 — Geometry detail & world scale ⬜

Surface and world-building detail the renderer currently can't express.

- ⬜ **Decals** — projector-box decals composited in a deferred-style screen pass (or
      mesh-clipped for the forward path), reading the depth buffer to wrap onto opaque
      geometry — bullet holes, blood, signage — without editing the target mesh.
- ⬜ **Parallax occlusion mapping** — a height-map ray-march in the fragment stage as an
      opt-in material flag (one more material-uniform bit), for deep brick/stone without
      the triangle count.
- ⬜ **Terrain / heightfield** — a `Terrain` node: GPU-clipmap or quadtree LOD heightfield
      with splat-mapped material layers, built on the existing LOD + instancing primitives.
- ✅ **Trails & ribbons** — `TrailRenderer` (a `Mesh`) samples `target`'s world position
      into a fixed-capacity history each `update(cameraPosition)` and rebuilds a camera-facing
      triangle strip through it (central-difference tangent × view → ribbon side), with width
      taper (`width`→`widthTail`) and length-wise UVs. Reuses the standard mesh path (any
      `Material`); spare index quads stay degenerate so the draw count is constant and
      allocation-free per frame. For sword arcs, tracers, motion streaks. Offline-verified
      (strip positions, UVs, billboard normal, taper, ring buffer, degenerate fill).

## Beyond — under consideration ⬜

Bigger bets that need a design pass before they earn a tier. Listed so the trajectory is
visible, not committed.

- ⬜ **WebXR** — a stereo/multiview render path (instanced two-eye draw) and XR input,
      the largest deviation from the single-camera assumption baked through the renderer.
- ⬜ **GPU-driven scene submission** — push culling → indirect draw further toward a fully
      GPU-built draw list (per-meshlet culling) once WebGPU exposes the needed primitives.
- ⬜ **Node/clip-graph editor surface for agents** — not a visual editor (explicitly out
      of scope), but a structured, `describe()`-style read/write API over post chains and
      animation graphs so an agent can edit the pipeline as data and diff the result.

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
7. **The engine stays generic — no use cases in `src/`.** The engine ships general,
   composable primitives and knows nothing about any specific look, art style, game, or
   asset. Cel/comic shading, outlines, a named avatar's face, water, terrain — these are
   *applications* and live in app code (e.g. `examples/gltf-viewer/comic.ts`), built on
   `ShaderMaterial` / `ShaderPass` / `ComputeTask`. No shader branch keyed to an asset,
   no renderer flag named after a look (`celShading`, `outline*`, `toon*`, `anime*`), no
   magic numbers tuned for one scene. If an app can't express a use case, that is an API
   gap: surface it and fill it with a *standard, general* capability any app could use —
   never by teaching the engine the use case. See docs/ARCHITECTURE.md → "The engine
   stays generic."
8. **Colors are structured values, not CSS hex.** Public color inputs take a `Color`
   (or a plain `vec3`/`vec4`/number-array) — explicit channels in a known color space —
   never a packed `0xRRGGBB` integer. Hex hides the color space and isn't introspectable
   as data; `Color.setHex()` stays available as an opt-in sRGB convenience, but it is not
   the API surface. New color-touching work (color grading, lights, materials) must
   accept and report colors this way.
