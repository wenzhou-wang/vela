import {
  AgentGraphEditor,
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  GridHelper,
  Mesh,
  OrbitControls,
  PerspectiveCamera,
  PointLight,
  Scene,
  ShaderPass,
  StandardMaterial,
  Vector3,
  WebGPURenderer,
  describeScene,
  type AgentGraphChange,
} from 'vela';

const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!;
if (!WebGPURenderer.isSupported()) {
  document.querySelector<HTMLElement>('#unsupported')!.style.display = 'grid';
  throw new Error('WebGPU not supported');
}

const renderer = new WebGPURenderer({ canvas, sampleCount: 1, pixelRatio: 1 });
await renderer.init();
renderer.postProcessing = true;
renderer.toneMapping = 'agx';
renderer.bloom = true;
renderer.bloomThreshold = 0.9;
renderer.bloomIntensity = 0.55;
renderer.gpuCulling = true;
renderer.gpuSceneSubmission = true;
renderer.clusteredLighting = true;
renderer.vignette = 0.2;

const contour = new ShaderPass({
  name: 'depth-contour',
  inputs: ['linearDepth', 'normal'],
  uniforms: { spacing: 2.5, strength: 0.75 },
  effect: /* wgsl */ `
    fn effect(uv : vec2<f32>) -> vec4<f32> {
      let source = sceneColor(uv);
      let depth = sceneLinearDepth(uv);
      let normal = sceneWorldNormal(uv);
      let geometry = select(1.0, 0.0, sceneDepth(uv) >= 0.9999);
      let line = pow(1.0 - abs(sin(depth * 3.14159265 / u.spacing)), 18.0);
      let facing = 0.35 + 0.65 * abs(normal.y);
      let ink = vec3<f32>(0.08, 1.0, 0.62) * line * facing * u.strength * geometry;
      return vec4<f32>(source.rgb + ink, source.a);
    }
  `,
});

const pulse = new ShaderPass({
  name: 'radar-pulse',
  inputs: ['linearDepth'],
  uniforms: { strength: 0.5 },
  effect: /* wgsl */ `
    fn effect(uv : vec2<f32>) -> vec4<f32> {
      let source = sceneColor(uv);
      let depth = sceneLinearDepth(uv);
      let geometry = select(1.0, 0.0, sceneDepth(uv) >= 0.9999);
      let wave = pow(max(0.0, sin(depth * 0.8 - pp.time.x * 2.5)), 20.0) * geometry;
      return vec4<f32>(source.rgb + vec3<f32>(0.15, 0.35, 1.0) * wave * u.strength, source.a);
    }
  `,
});
renderer.passes.push(contour, pulse);
const graph = new AgentGraphEditor(renderer);

const scene = new Scene();
scene.background = new Color().setHex(0x04100d);
scene.fog = { color: new Color().setHex(0x04100d), density: 0.014 };
scene.sky = { sunDirection: new Vector3(0.4, 0.6, -0.5), turbidity: 7 };
scene.environmentIntensity = 0.3;

const camera = new PerspectiveCamera(52, 1, 0.1, 180);
camera.position.set(18, 15, 25);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 3.5, 0);
controls.enableDamping = true;

const sun = new DirectionalLight(new Color().setHex(0xc9ffe9), 2.2);
sun.position.set(12, 20, 9);
scene.add(sun, sun.target, new AmbientLight(new Color().setHex(0x24473d), 0.5));

const geometry = new BoxGeometry(1.25, 1, 1.25);
geometry.computeMeshlets(4);
const material = new StandardMaterial({
  color: new Color().setHex(0x294d43),
  metalness: 0.7,
  roughness: 0.3,
});
for (let i = 0; i < 48; i++) {
  const angle = i * 0.52;
  const radius = 2.5 + i * 0.115;
  const tower = new Mesh(geometry, material);
  tower.position.set(Math.cos(angle) * radius, 0.3 + (i % 9) * 0.55, Math.sin(angle) * radius);
  tower.scale.y = 0.7 + (i % 7) * 0.48;
  tower.rotation.y = -angle;
  if (i % 12 === 0) tower.name = `telemetry-tower-${i / 12 + 1}`;
  scene.add(tower);
}

for (let i = 0; i < 8; i++) {
  const light = new PointLight(
    new Color().setHex(i % 2 ? 0x3c7dff : 0x42ffae),
    22,
    8,
  );
  const angle = i / 8 * Math.PI * 2;
  light.position.set(Math.cos(angle) * 8, 2.2 + (i % 3) * 2, Math.sin(angle) * 8);
  light.name = `cluster-light-${i + 1}`;
  scene.add(light);
}

const grid = new GridHelper(60, 60, new Color().setHex(0x1e6b52), new Color().setHex(0x0b211b));
grid.position.y = -0.25;
scene.add(grid);

const readout = document.querySelector<HTMLElement>('#readout')!;
const status = document.querySelector<HTMLElement>('#status')!;
let lastDiff: AgentGraphChange[] = [];
let mode = 0;
let view: 'telemetry' | 'scene' | 'graph' = 'telemetry';

document.querySelector<HTMLButtonElement>('#mutate')!.addEventListener('click', () => {
  mode = (mode + 1) % 3;
  if (mode === 0) {
    lastDiff = graph.apply([
      { type: 'post.update', id: contour.id, patch: { enabled: true, uniforms: { spacing: 2.5, strength: 0.75 } } },
      { type: 'post.update', id: pulse.id, patch: { enabled: true } },
      { type: 'post.move', id: contour.id, index: 0 },
    ]);
  } else if (mode === 1) {
    lastDiff = graph.apply([
      { type: 'post.update', id: contour.id, patch: { uniforms: { spacing: 0.8, strength: 1.25 } } },
      { type: 'post.update', id: pulse.id, patch: { enabled: false } },
    ]);
  } else {
    lastDiff = graph.apply([
      { type: 'post.update', id: contour.id, patch: { enabled: false } },
      { type: 'post.update', id: pulse.id, patch: { enabled: true, uniforms: { strength: 1.15 } } },
      { type: 'post.move', id: pulse.id, index: 0 },
    ]);
  }
  status.textContent = `Applied ${lastDiff.length} structural changes through AgentGraphEditor`;
  view = 'graph';
  updateReadout();
});

document.querySelector<HTMLButtonElement>('#diagnose')!.addEventListener('click', () => {
  const findings = renderer.diagnose(scene, camera);
  status.textContent = findings.length ? `${findings.length} diagnostics found` : 'Diagnostics clean: no suspicious configuration found';
  readout.textContent = JSON.stringify({ diagnostics: findings }, null, 2);
});

document.querySelector<HTMLButtonElement>('#view')!.addEventListener('click', () => {
  view = view === 'telemetry' ? 'scene' : view === 'scene' ? 'graph' : 'telemetry';
  status.textContent = `Viewing ${view}`;
  updateReadout();
});

function updateReadout(): void {
  if (view === 'scene') {
    readout.textContent = JSON.stringify({ scene: describeScene(scene, camera) }, null, 2);
    return;
  }
  if (view === 'graph') {
    const snapshot = graph.describe();
    readout.textContent = JSON.stringify({
      postGraph: snapshot.post.map(({ id, name, enabled, inputs, uniforms }) => ({ id, name, enabled, inputs, uniforms })),
      lastDiff,
    }, null, 2);
    return;
  }
  readout.textContent = JSON.stringify({ render: renderer.report() }, null, 2);
}

function resize(): void {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

let lastTelemetry = 0;
function animate(now: number): void {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  if (view === 'telemetry' && now - lastTelemetry > 500) {
    updateReadout();
    lastTelemetry = now;
  }
}
requestAnimationFrame(animate);
