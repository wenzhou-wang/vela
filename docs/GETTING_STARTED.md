# Getting started

## Install & run the viewer

```bash
npm install
npm run dev        # opens examples/gltf-viewer in your browser
```

Requires a WebGPU-capable browser (Chrome/Edge 113+, Safari 18, or Firefox with WebGPU
enabled). Build a production bundle with `npm run build`; type-check with `npm run typecheck`.

## Your first scene

vela's API is intentionally close to three.js, so the mental model transfers. The one
difference that matters: **the renderer is initialized asynchronously**, because it has to
request a GPU adapter and device.

```ts
import {
  WebGPURenderer, Scene, PerspectiveCamera, OrbitControls,
  Mesh, SphereGeometry, StandardMaterial,
  DirectionalLight, AmbientLight, Color,
} from 'vela';

const canvas = document.querySelector('canvas')!;

// 1. Renderer — note the await
const renderer = new WebGPURenderer({ canvas, sampleCount: 4 });
await renderer.init();

// 2. Scene + camera
const scene = new Scene();
scene.background = new Color().setHex(0x10131a);

const camera = new PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
camera.position.set(0, 1, 4);

const controls = new OrbitControls(camera, canvas);

// 3. A PBR mesh
const mesh = new Mesh(
  new SphereGeometry(1, 48, 32),
  new StandardMaterial({ color: new Color().setHex(0x4f9dff), metalness: 0.2, roughness: 0.35 }),
);
scene.add(mesh);

// 4. Lights
const key = new DirectionalLight(new Color().setHex(0xffffff), 3);
key.position.set(5, 8, 6);
scene.add(key, key.target);
scene.add(new AmbientLight(new Color().setHex(0x6677aa), 0.4));

// 5. Render loop
function frame() {
  requestAnimationFrame(frame);
  controls.update();
  renderer.render(scene, camera);
}
frame();
```

## Loading a glTF model

```ts
import { GLTFLoader } from 'vela';

const loader = new GLTFLoader();
const { scene: model, boundingBox, materials } = await loader.load('/models/helmet.glb');
scene.add(model);

// Frame the camera to the model
import { Vector3 } from 'vela';
const center = new Vector3(), size = new Vector3();
boundingBox.getCenter(center);
boundingBox.getSize(size);
const dist = Math.max(size.x, size.y, size.z) * 1.6;
camera.position.copy(center).add(new Vector3(0.6, 0.4, 1).normalize().multiplyScalar(dist));
controls.setTarget(center, dist);
```

`GLTFLoader.parse(arrayBuffer)` is also available when you already have the bytes (e.g. a
dropped file). `.glb` is fully self-contained; for `.gltf` with external `.bin`/textures,
rewrite the relative URIs to object URLs first — see `examples/gltf-viewer/main.ts`.

## Handling resize

```ts
function onResize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h);          // CSS pixels; pixel ratio applied internally
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
onResize();
```

## Tips

- **Exposure / tonemapping**: `renderer.exposure = 1.5` (ACES filmic is applied in-shader).
- **Pixel ratio**: capped at 2 by default; override with `new WebGPURenderer({ ..., pixelRatio })`
  or `renderer.setPixelRatio(n)`.
- **Transparency**: set `material.transparent = true`; transparent meshes are drawn after
  opaque ones, back-to-front.
- **Double-sided**: `material.side = 'double'`. **Alpha cutout**: `material.alphaTest = 0.5`.
- **Colors**: every color input takes a `Color` or a linear `[r, g, b]` array — never a
  packed `0xRRGGBB` integer. For sRGB hex, opt in explicitly: `new Color().setHex(0x4f9dff)`.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for how the renderer works under the hood, and
[../ROADMAP.md](../ROADMAP.md) for what's planned.
