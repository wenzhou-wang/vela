import {
  WebGPURenderer,
  Scene,
  PerspectiveCamera,
  OrbitControls,
  InstancedMesh,
  Mesh,
  BoxGeometry,
  PlaneGeometry,
  StandardMaterial,
  DirectionalLight,
  AmbientLight,
  Color,
  Vector3,
  Quaternion,
  Matrix4,
  Stats,
} from 'vela';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

if (!WebGPURenderer.isSupported()) {
  (document.getElementById('unsupported') as HTMLElement).style.display = 'flex';
  throw new Error('WebGPU not supported');
}

// ---- Renderer / scene / camera ----
const renderer = new WebGPURenderer({ canvas });
await renderer.init();
renderer.shadows = true;
renderer.toneMapping = 'aces';

const scene = new Scene();
scene.sky = { sunDirection: new Vector3(0.4, 0.22, -0.5), turbidity: 3 };
scene.skybox = true;
scene.environmentIntensity = 0.7;
scene.fog = { color: new Color().setHex(0x0a0c14), density: 0.012 };

const camera = new PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 500);
camera.position.set(0, 14, 26);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 1.5, 0);

// ---- Lights ----
const sun = new DirectionalLight(new Color().setHex(0xfff1d6), 3.2);
sun.position.set(18, 26, 12);
sun.castShadow = true;
scene.add(sun, sun.target);

const fill = new DirectionalLight(new Color().setHex(0x6fa8ff), 0.8);
fill.position.set(-14, 8, -10);
scene.add(fill, fill.target);

scene.add(new AmbientLight(new Color().setHex(0x33405a), 0.6));

// ---- Floor (catches shadows) ----
const floor = new Mesh(
  new PlaneGeometry(80, 80),
  new StandardMaterial({ color: new Color().setHex(0x0c0e16), roughness: 0.95, metalness: 0 }),
);
scene.add(floor);

// ---- Instanced pillar grid ----
const GRID = 64; // GRID*GRID pillars in one draw call
const COUNT = GRID * GRID;
const SPACING = 0.5;
const EXTENT = (GRID - 1) * SPACING; // world width of the field

const pillar = new BoxGeometry(0.34, 1, 0.34); // unit-tall; we scale Y per frame
const pillarMat = new StandardMaterial({
  color: new Color().setHex(0xffb347),
  emissive: new Color().setHex(0x4a2a00),
  emissiveIntensity: 1,
  metalness: 0.9,
  roughness: 0.28,
});
const field = new InstancedMesh(pillar, pillarMat, COUNT);
scene.add(field);

(document.getElementById('count') as HTMLElement).textContent = COUNT.toLocaleString();

// Per-instance grid position in XZ (height is computed each frame).
const baseX = new Float32Array(COUNT);
const baseZ = new Float32Array(COUNT);
for (let j = 0; j < GRID; j++) {
  for (let i = 0; i < GRID; i++) {
    const idx = j * GRID + i;
    baseX[idx] = i * SPACING - EXTENT / 2;
    baseZ[idx] = j * SPACING - EXTENT / 2;
  }
}

// ---- Controls state ----
const ui = {
  sources: 3,
  speed: 1,
  freq: 1.6,
  orbit: true,
};
const bind = (id: string, key: keyof typeof ui, parse: (v: string) => number | boolean) => {
  const el = document.getElementById(id) as HTMLInputElement;
  const apply = () => ((ui[key] as number | boolean) = parse(el.type === 'checkbox' ? String(el.checked) : el.value));
  el.addEventListener('input', apply);
  apply();
};
bind('sources', 'sources', (v) => Math.round(Number(v)));
bind('speed', 'speed', (v) => Number(v));
bind('freq', 'freq', (v) => Number(v));
bind('orbit', 'orbit', (v) => v === 'true');

// ---- Wave evaluation ----
// h(p,t) = Σ A · sin(k·|p - sᵢ| - ω·t) / (1 + |p - sᵢ|)
// Sources drift along Lissajous orbits so the interference pattern is never static.
const MAX_SOURCES = 6;
const srcX = new Float32Array(MAX_SOURCES);
const srcZ = new Float32Array(MAX_SOURCES);

function updateSources(t: number): void {
  const r = EXTENT * 0.42;
  for (let s = 0; s < MAX_SOURCES; s++) {
    const a = 0.7 + s * 0.35;
    const b = 0.5 + s * 0.27;
    const phase = (s / MAX_SOURCES) * Math.PI * 2;
    srcX[s] = Math.cos(t * a + phase) * r;
    srcZ[s] = Math.sin(t * b + phase * 1.3) * r;
  }
}

// ---- Per-frame matrix update ----
const pos = new Vector3();
const scl = new Vector3();
const quat = new Quaternion(); // identity
const mat = new Matrix4();

function updateField(t: number): void {
  const n = ui.sources;
  const k = ui.freq;
  const omega = 2.2;
  const amp = 0.9;

  for (let idx = 0; idx < COUNT; idx++) {
    const px = baseX[idx];
    const pz = baseZ[idx];

    let h = 0;
    for (let s = 0; s < n; s++) {
      const dx = px - srcX[s];
      const dz = pz - srcZ[s];
      const d = Math.sqrt(dx * dx + dz * dz);
      h += (amp * Math.sin(k * d - omega * t)) / (1 + 0.25 * d);
    }
    // Map signed wave height to a positive pillar height (always >= 0.05).
    const height = 0.6 + h * 1.6;
    const sy = Math.max(0.05, height);

    pos.set(px, sy * 0.5, pz); // BoxGeometry is centered; lift so the base sits on y=0
    scl.set(1, sy, 1);
    mat.compose(pos, quat, scl);
    field.setMatrixAt(idx, mat);
  }
  field.needsUpdate();
}

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

// ---- Stats overlay ----
const stats = new Stats();
if (stats.dom) {
  stats.dom.style.position = 'absolute';
  stats.dom.style.top = '16px';
  stats.dom.style.right = '16px';
  document.body.appendChild(stats.dom);
}

// ---- Render loop ----
let last = performance.now();
let simTime = 0;

function animate(): void {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  simTime += dt * ui.speed;

  updateSources(simTime);
  updateField(simTime);

  // Auto-orbit via the controls' own rotation — setting camera.position directly
  // would just be overwritten by controls.update() each frame.
  controls.autoRotate = ui.orbit;
  controls.update();
  renderer.render(scene, camera);
  stats.update();
}
animate();
