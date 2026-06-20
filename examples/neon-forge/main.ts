import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  GridHelper,
  InstancedMesh,
  Matrix4,
  Mesh,
  OrbitControls,
  ParticleSystem,
  PerspectiveCamera,
  Quaternion,
  Scene,
  ShaderMaterial,
  ShaderPass,
  SphereGeometry,
  StandardMaterial,
  Stats,
  TrailRenderer,
  Vector3,
  WebGPURenderer,
} from 'vela';

const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!;
if (!WebGPURenderer.isSupported()) {
  document.querySelector<HTMLElement>('#unsupported')!.style.display = 'grid';
  throw new Error('WebGPU not supported');
}

const renderer = new WebGPURenderer({ canvas, sampleCount: 1 });
await renderer.init();
renderer.postProcessing = true;
renderer.toneMapping = 'agx';
renderer.bloom = true;
renderer.bloomThreshold = 0.65;
renderer.bloomIntensity = 1.1;
renderer.autoExposure = true;
renderer.autoExposureMinEV = -3;
renderer.autoExposureMaxEV = 1;
renderer.chromaticAberration = 0.0015;
renderer.vignette = 0.28;

const signalWarp = new ShaderPass({
  name: 'signal-warp',
  uniforms: { amount: 0.0025 },
  effect: /* wgsl */ `
    fn effect(uv : vec2<f32>) -> vec4<f32> {
      let scan = sin(uv.y * pp.resolution.y * 1.4 + pp.time * 8.0);
      let offset = vec2<f32>(scan * u.amount, 0.0);
      let base = sceneColor(uv + offset);
      let glow = sceneColor(uv - offset * 2.0);
      return vec4<f32>(base.r, mix(base.g, glow.g, 0.22), glow.b, base.a);
    }
  `,
});
renderer.passes.push(signalWarp);

const scene = new Scene();
scene.fog = { color: new Color().setHex(0x030511), density: 0.016 };
scene.sky = { sunDirection: new Vector3(-0.3, 0.65, -0.4), turbidity: 9 };
scene.environmentIntensity = 0.18;

const camera = new PerspectiveCamera(55, 1, 0.1, 220);
camera.position.set(0, 11, 24);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 3, 0);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.55;

const key = new DirectionalLight(new Color().setHex(0x90eaff), 1.8);
key.position.set(-8, 14, 5);
scene.add(key, key.target, new AmbientLight(new Color().setHex(0x312b62), 0.42));

const sculptureMaterial = new ShaderMaterial({
  name: 'reactive-chrome',
  uniforms: { energy: 1 },
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      let bands = 0.5 + 0.5 * sin(in.worldPos.y * 2.6 - elapsedTime() * 3.0);
      let edge = pow(1.0 - abs(normalize(in.worldNormal).y), 2.0);
      s.baseColor = mix(vec3<f32>(0.015, 0.025, 0.05), vec3<f32>(0.08, 0.16, 0.22), bands);
      s.metalness = 0.92;
      s.roughness = 0.2;
      s.emissive = mix(vec3<f32>(0.0, 0.35, 0.75), vec3<f32>(0.9, 0.0, 0.55), bands) * edge * u.energy;
      return s;
    }
  `,
});

const COUNT = 120;
const blocks = new InstancedMesh(new BoxGeometry(0.42, 1.8, 0.42), sculptureMaterial, COUNT);
blocks.name = 'energy-helix';
const matrix = new Matrix4();
const position = new Vector3();
const scale = new Vector3();
const rotation = new Quaternion();
for (let i = 0; i < COUNT; i++) {
  const t = i / COUNT * Math.PI * 8;
  const radius = 4.1 + Math.sin(t * 1.5) * 0.55;
  position.set(Math.cos(t) * radius, 0.3 + i / COUNT * 8, Math.sin(t) * radius);
  scale.set(1, 0.45 + (Math.sin(t * 2.3) * 0.5 + 0.5) * 1.1, 1);
  rotation.setFromAxisAngle(new Vector3(0, 1, 0), -t);
  matrix.compose(position, rotation, scale);
  blocks.setMatrixAt(i, matrix);
}
blocks.needsUpdate();
scene.add(blocks);

const grid = new GridHelper(70, 70, new Color().setHex(0x17445e), new Color().setHex(0x091827));
grid.position.y = -0.8;
scene.add(grid);

const colors = [0x43f4ff, 0xff3ebf, 0x9b70ff];
const cores: Mesh[] = [];
const emitters: ParticleSystem[] = [];
const trails: TrailRenderer[] = [];
for (let i = 0; i < colors.length; i++) {
  const color = new Color().setHex(colors[i]);
  const core = new Mesh(new SphereGeometry(0.36, 20, 12), new StandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 8,
    roughness: 0.16,
  }));
  core.name = `orbiting-core-${i + 1}`;
  const emitter = new ParticleSystem({
    capacity: 1600,
    rate: 430,
    lifetime: [0.45, 1.35],
    velocity: new Vector3(0, 0.75, 0),
    spread: 1.2,
    gravity: new Vector3(0, -0.8, 0),
    size: [0.16, 0.015],
    color: [color, new Color().setHex(0x101840)],
    opacity: [1, 0],
  });
  const trail = new TrailRenderer({
    maxPoints: 72,
    width: 0.34,
    widthTail: 0.01,
    minDistance: 0.045,
    material: new StandardMaterial({ color, emissive: color, emissiveIntensity: 5, side: 'double' }),
  });
  trail.target = core;
  cores.push(core);
  emitters.push(emitter);
  trails.push(trail);
  scene.add(core, emitter, trail);
}

const energyInput = document.querySelector<HTMLInputElement>('#energy')!;
const bloomInput = document.querySelector<HTMLInputElement>('#bloom')!;
const warpInput = document.querySelector<HTMLInputElement>('#warp')!;
bloomInput.addEventListener('input', () => { renderer.bloomIntensity = Number(bloomInput.value); });
warpInput.addEventListener('input', () => { signalWarp.enabled = warpInput.checked; });

const stats = new Stats();
if (stats.dom) {
  stats.dom.style.cssText = 'position:fixed;top:18px;right:18px';
  document.body.append(stats.dom);
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

function animate(now: number): void {
  requestAnimationFrame(animate);
  const energy = Number(energyInput.value);
  const t = now * 0.001 * energy;
  sculptureMaterial.uniforms.energy = energy;
  blocks.rotation.y = t * 0.1;
  for (let i = 0; i < cores.length; i++) {
    const phase = t * (0.7 + i * 0.12) + i * Math.PI * 2 / 3;
    const radius = 6.2 + Math.sin(t * 0.8 + i) * 1.1;
    const x = Math.cos(phase) * radius;
    const y = 3.8 + Math.sin(phase * 1.7 + i) * 3.2;
    const z = Math.sin(phase) * radius;
    cores[i].position.set(x, y, z);
    emitters[i].position.set(x, y, z);
    emitters[i].options.rate = 430 * energy;
    trails[i].update(camera.position);
  }
  controls.update();
  renderer.render(scene, camera);
  stats.update();
}
requestAnimationFrame(animate);
