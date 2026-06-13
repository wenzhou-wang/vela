import {
  WebGPURenderer,
  Scene,
  PerspectiveCamera,
  OrbitControls,
  GLTFLoader,
  DirectionalLight,
  AmbientLight,
  PointLight,
  Object3D,
  Mesh,
  InstancedMesh,
  BoxGeometry,
  SphereGeometry,
  PlaneGeometry,
  StandardMaterial,
  Texture,
  Color,
  Box3,
  Vector3,
  Matrix4,
  Quaternion,
  AnimationMixer,
  type GLTFResult,
} from 'vela';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;

if (!WebGPURenderer.isSupported()) {
  (document.getElementById('unsupported') as HTMLElement).style.display = 'flex';
  throw new Error('WebGPU not supported');
}

// Single-sample depth is required by the cel outline pass; FXAA handles final edges.
const renderer = new WebGPURenderer({ canvas, sampleCount: 1 });
await renderer.init();

const scene = new Scene();
scene.background = new Color().setHex(0x10131a);
scene.ambientColor = new Color().setHex(0x404a5a);
scene.ambientIntensity = 1.0;

const camera = new PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 1000);
camera.position.set(3, 2, 5);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// ---- Lighting rig ----
const key = new DirectionalLight(0xfff4e6, 3.0);
key.position.set(5, 8, 6);
scene.add(key, key.target);

const fill = new DirectionalLight(0xa9c7ff, 1.1);
fill.position.set(-6, 3, -4);
scene.add(fill, fill.target);

const rim = new PointLight(0xffffff, 12, 30, 2);
rim.position.set(-3, 4, -5);
scene.add(rim);

const ambient = new AmbientLight(0x6677aa, 0.4);
scene.add(ambient);

// ---- Default showcase scene (until a model is loaded) ----
let currentModel: Object3D | null = null;

function checkerTexture(): Texture {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d')!;
  const tile = 32;
  for (let y = 0; y < size; y += tile) {
    for (let x = 0; x < size; x += tile) {
      const on = ((x / tile) + (y / tile)) % 2 === 0;
      ctx.fillStyle = on ? '#3a3f4b' : '#2a2e38';
      ctx.fillRect(x, y, tile, tile);
    }
  }
  const tex = new Texture(c as unknown as ImageBitmap, { colorSpace: 'srgb' });
  tex.needsUpdate();
  return tex;
}

function buildShowcase(): Object3D {
  const group = new Object3D();

  const ground = new Mesh(
    new PlaneGeometry(40, 40),
    new StandardMaterial({ map: checkerTexture(), roughness: 0.95, metalness: 0.0 }),
  );
  ground.position.y = -1.0;
  group.add(ground);

  const rows = 5;
  const cols = 7;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const metalness = i / (rows - 1);
      const roughness = Math.max(0.05, j / (cols - 1));
      const sphere = new Mesh(
        new SphereGeometry(0.42, 48, 32),
        new StandardMaterial({ color: 0x9aa7ff, metalness, roughness }),
      );
      sphere.position.set((j - (cols - 1) / 2) * 1.1, (i - (rows - 1) / 2) * 1.1 + 0.4, 0);
      group.add(sphere);
    }
  }
  return group;
}

function frameObject(box: Box3): void {
  const center = new Vector3();
  const size = new Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dist = (maxDim / 2) / Math.tan((camera.fov * Math.PI) / 360) * 1.6;

  camera.near = Math.max(maxDim / 1000, 0.001);
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();

  camera.position.copy(center).add(new Vector3(0.6, 0.4, 1).normalize().multiplyScalar(dist));
  controls.setTarget(center, dist);
}

function setModel(object: Object3D, box: Box3): void {
  if (currentModel) scene.remove(currentModel);
  currentModel = object;
  scene.add(object);
  frameObject(box);
  // aim key/fill lights at the model center
  const center = new Vector3();
  box.getCenter(center);
  key.target.position.copy(center);
  fill.target.position.copy(center);
}

// initial scene
{
  const showcase = buildShowcase();
  showcase.updateMatrixWorld(true);
  const box = new Box3();
  const tmp = new Box3();
  showcase.traverse((o) => {
    if (o instanceof Mesh) {
      o.geometry.computeBoundingBox();
      if (o.geometry.boundingBox) {
        tmp.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld);
        box.union(tmp);
      }
    }
  });
  setModel(showcase, box);
}

// ---- Loading ----
const loader = new GLTFLoader();
const status = (msg: string) => { statusMsg = msg; };
let statusMsg = '';

async function loadFromURL(url: string, isBitmoji = false): Promise<void> {
  setBitmojiSelected(false);
  status('Loading…');
  try {
    const result = await loader.load(url);
    onLoaded(result, isBitmoji);
  } catch (e) {
    console.error(e);
    status(`Error: ${(e as Error).message}`);
  }
}

async function loadFromFiles(files: File[]): Promise<void> {
  const byName = new Map<string, File>();
  for (const f of files) byName.set(f.name, f);

  const main = files.find((f) => /\.(glb|gltf)$/i.test(f.name));
  if (!main) { status('No .glb/.gltf in drop'); return; }

  setBitmojiSelected(false);
  status('Loading…');
  try {
    if (/\.glb$/i.test(main.name)) {
      const buf = await main.arrayBuffer();
      onLoaded(await loader.parse(buf));
      return;
    }
    // .gltf: rewrite relative URIs to object URLs for dropped sibling files
    const json = JSON.parse(await main.text());
    const urls: string[] = [];
    const resolve = (uri?: string) => {
      if (!uri || uri.startsWith('data:')) return uri;
      const file = byName.get(decodeURIComponent(uri.split('/').pop()!));
      if (!file) return uri;
      const u = URL.createObjectURL(file);
      urls.push(u);
      return u;
    };
    for (const b of json.buffers ?? []) b.uri = resolve(b.uri);
    for (const im of json.images ?? []) if (im.uri) im.uri = resolve(im.uri);
    const encoded = new TextEncoder().encode(JSON.stringify(json));
    onLoaded(await loader.parse(encoded.buffer as ArrayBuffer));
    urls.forEach((u) => URL.revokeObjectURL(u));
  } catch (e) {
    console.error(e);
    status(`Error: ${(e as Error).message}`);
  }
}

let mixer: AnimationMixer | null = null;
let clips: GLTFResult['animations'] = [];

function onLoaded(result: GLTFResult, isBitmoji = false): void {
  setModel(result.scene, result.boundingBox);
  // Bitmoji defaults to the comic style; the dropdown offers PBR/comic/anime.
  setBitmojiSelected(isBitmoji, isBitmoji ? 'comic' : 'pbr');
  setupAnimations(result.animations);
  const extra = result.animations.length ? ` · ${result.animations.length} anim` : '';
  status(`Loaded · ${result.materials.length} materials${extra}`);
}

function setupAnimations(animations: GLTFResult['animations']): void {
  mixer = null;
  clips = animations;
  const row = document.getElementById('animRow') as HTMLElement;
  const select = document.getElementById('animSelect') as HTMLSelectElement;
  select.innerHTML = '';
  if (!animations.length) { row.style.display = 'none'; return; }

  row.style.display = 'flex';
  animations.forEach((clip, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = clip.name || `clip ${i}`;
    select.appendChild(opt);
  });
  mixer = new AnimationMixer().play(animations[0]);
  select.value = '0';
}

document.getElementById('animSelect')!.addEventListener('change', (e) => {
  const i = parseInt((e.target as HTMLSelectElement).value, 10);
  if (clips[i]) mixer = new AnimationMixer().play(clips[i]);
});

// ---- UI wiring ----
const exposureEl = document.getElementById('exposure') as HTMLInputElement;
exposureEl.addEventListener('input', () => { renderer.exposure = parseFloat(exposureEl.value); });

const lightEl = document.getElementById('lightIntensity') as HTMLInputElement;
lightEl.addEventListener('input', () => { key.intensity = parseFloat(lightEl.value); });

const styleEl = document.getElementById('renderStyle') as HTMLSelectElement;
const outlineEl = document.getElementById('outlineThickness') as HTMLInputElement;
const styleRow = document.getElementById('styleRow') as HTMLLabelElement;
const outlineRow = document.getElementById('outlineRow') as HTMLLabelElement;
// 'pbr' (default), 'comic' (cel + bold outlines), 'anime' (toon ramp + ink lines).
function setRenderStyle(style: string): void {
  const comic = style === 'comic';
  const anime = style === 'anime';
  renderer.celShading = comic;
  renderer.animeShading = anime;
  renderer.postProcessing = comic || anime;
  // Comic uses bolder lines; anime keeps them thinner.
  if (comic || anime) outlineEl.value = comic ? '2' : '1.25';
  renderer.outlineThickness = parseFloat(outlineEl.value);
  outlineEl.disabled = !(comic || anime);
}
function setBitmojiSelected(selected: boolean, defaultStyle = 'pbr'): void {
  styleRow.style.display = selected ? 'flex' : 'none';
  outlineRow.style.display = selected ? 'flex' : 'none';
  styleEl.disabled = !selected;
  styleEl.value = selected ? defaultStyle : 'pbr';
  setRenderStyle(styleEl.value);
}
styleEl.addEventListener('change', () => setRenderStyle(styleEl.value));
outlineEl.addEventListener('input', () => {
  renderer.outlineThickness = parseFloat(outlineEl.value);
});

const autoEl = document.getElementById('autorotate') as HTMLInputElement;
autoEl.addEventListener('change', () => { controls.autoRotate = autoEl.checked; });

const backgroundEl = document.getElementById('backgroundMode') as HTMLSelectElement;
backgroundEl.addEventListener('change', () => {
  scene.background!.setHex(backgroundEl.value === 'light' ? 0xe8edf5 : 0x10131a);
});

document.getElementById('reset')!.addEventListener('click', () => {
  if (currentModel) {
    const box = new Box3();
    const tmp = new Box3();
    currentModel.updateMatrixWorld(true);
    currentModel.traverse((o) => {
      if (o instanceof Mesh && o.geometry.boundingBox) {
        tmp.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld);
        box.union(tmp);
      }
    });
    if (!box.isEmpty()) frameObject(box);
  }
});

const fileInput = document.getElementById('fileInput') as HTMLInputElement;
document.getElementById('loadFile')!.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files.length) loadFromFiles([...fileInput.files]);
});

document.querySelectorAll<HTMLButtonElement>('.sample').forEach((btn) => {
  btn.addEventListener('click', () => {
    loadFromURL(btn.dataset.url!, btn.dataset.cel === 'true');
  });
});

// Instancing demo: a rippling field of cubes drawn in a single draw call.
document.getElementById('instDemo')!.addEventListener('click', () => {
  setBitmojiSelected(false);
  setupAnimations([]);
  const side = 40;
  const n = side * side;
  const group = new Object3D();
  const inst = new InstancedMesh(
    new BoxGeometry(0.55, 0.55, 0.55),
    new StandardMaterial({ color: 0x88aaff, metalness: 0.7, roughness: 0.3 }),
    n,
  );
  const m = new Matrix4();
  const pos = new Vector3();
  const q = new Quaternion();
  const scl = new Vector3(1, 1, 1);
  const axis = new Vector3(0, 1, 0);
  let i = 0;
  for (let x = 0; x < side; x++) {
    for (let z = 0; z < side; z++) {
      const fx = x - (side - 1) / 2;
      const fz = z - (side - 1) / 2;
      const h = Math.sin(fx * 0.45) * Math.cos(fz * 0.45) * 2.5;
      pos.set(fx * 1.1, h, fz * 1.1);
      q.setFromAxisAngle(axis, (x + z) * 0.18);
      m.compose(pos, q, scl);
      inst.setMatrixAt(i++, m);
    }
  }
  inst.needsUpdate();
  group.add(inst);

  const half = (side * 1.1) / 2 + 1;
  const box = new Box3();
  box.expandByPoint(new Vector3(-half, -3, -half));
  box.expandByPoint(new Vector3(half, 3, half));
  setModel(group, box);
  status(`Instanced · ${n.toLocaleString()} cubes · 1 draw call`);
});

// drag & drop
window.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
window.addEventListener('dragleave', (e) => { if (e.relatedTarget === null) document.body.classList.remove('dragging'); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('dragging');
  if (e.dataTransfer?.files.length) loadFromFiles([...e.dataTransfer.files]);
});

// ---- Resize ----
function resize(): void {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---- Render loop ----
let frames = 0;
let lastFpsTime = performance.now();
let lastFrameTime = performance.now();
let fps = 0;

function animate(): void {
  requestAnimationFrame(animate);
  const now0 = performance.now();
  const dt = Math.min((now0 - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now0;

  if (mixer) {
    mixer.update(dt);
    currentModel?.updateMatrixWorld(true);
  }
  controls.update();
  renderer.render(scene, camera);

  frames++;
  const now = performance.now();
  if (now - lastFpsTime >= 500) {
    fps = Math.round((frames * 1000) / (now - lastFpsTime));
    frames = 0;
    lastFpsTime = now;
    const dpr = (window.devicePixelRatio || 1).toFixed(1);
    const culled = renderer.culledCount > 0 ? ` · ${renderer.culledCount} culled` : '';
    statsEl.innerHTML = `<b>${fps}</b> fps · ${renderer.drawingBufferWidth}×${renderer.drawingBufferHeight} · dpr ${dpr}${culled}${statusMsg ? ` · ${statusMsg}` : ''}`;
  }
}
animate();
