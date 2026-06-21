import {
  AmbientLight,
  Color,
  DirectionalLight,
  Mesh,
  OrbitControls,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  StandardMaterial,
  Stats,
  Terrain,
  Vector3,
  WebGPURenderer,
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
renderer.bloomThreshold = 1.05;
renderer.bloomIntensity = 0.35;
renderer.shadows = true;
renderer.shadowCascades = 2;
renderer.shadowMapSize = 1024;
renderer.volumetricFog = false;
renderer.vignette = 0.18;

const scene = new Scene();
const fogColor = new Color().setHex(0x817585);
scene.fog = { color: fogColor, density: 0.009, heightFalloff: 0.055 };
scene.sky = { sunDirection: new Vector3(0.55, 0.24, -0.55), turbidity: 5.5 };
scene.skybox = true;
scene.environmentIntensity = 0.48;

const camera = new PerspectiveCamera(52, 1, 0.2, 520);
camera.position.set(70, 32, 78);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 8, -8);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.28;

const sun = new DirectionalLight(new Color().setHex(0xffcf9e), 4.2);
sun.position.set(90, 48, -80);
sun.castShadow = true;
scene.add(sun, sun.target);
scene.add(new AmbientLight(new Color().setHex(0x56647f), 0.38));

const SEGMENTS = 96;
const heights = new Float32Array((SEGMENTS + 1) * (SEGMENTS + 1));
for (let z = 0; z <= SEGMENTS; z++) {
  for (let x = 0; x <= SEGMENTS; x++) {
    const nx = x / SEGMENTS * 2 - 1;
    const nz = z / SEGMENTS * 2 - 1;
    const ridge = Math.pow(Math.max(0, 1 - Math.abs(nx * 0.72 + nz * 0.3)), 2.2) * 15;
    const mountains = Math.max(0, nz * -0.7 + 0.2) * (12 + 10 * Math.sin(nx * 4.2));
    const detail = Math.sin(nx * 15 + Math.sin(nz * 8)) * 1.8 + Math.cos(nz * 17 - nx * 3) * 1.2;
    const lakeBed = Math.max(0, 1 - Math.hypot(nx + 0.25, nz - 0.3) * 2.2) * -7;
    heights[z * (SEGMENTS + 1) + x] = ridge + mountains + detail + lakeBed - 2;
  }
}

const terrainMaterial = new ShaderMaterial({
  name: 'altitude-and-slope',
  surface: /* wgsl */ `
    fn surface(in : VSOut) -> Surface {
      var s = defaultSurface(in);
      let n = normalize(in.worldNormal);
      let slope = 1.0 - max(n.y, 0.0);
      let grass = vec3<f32>(0.16, 0.22, 0.12);
      let rock = vec3<f32>(0.27, 0.23, 0.22);
      let snow = vec3<f32>(0.7, 0.72, 0.7);
      let ground = mix(grass, rock, smoothstep(0.2, 0.72, slope));
      s.baseColor = mix(ground, snow, smoothstep(15.0, 27.0, in.worldPos.y) * (1.0 - slope * 0.55));
      s.roughness = mix(0.88, 0.58, slope);
      s.metalness = 0.0;
      return s;
    }
  `,
});

const terrain = new Terrain({
  heights,
  segmentsX: SEGMENTS,
  segmentsZ: SEGMENTS,
  width: 180,
  depth: 180,
  tiles: [4, 4],
  levels: 3,
  material: terrainMaterial,
});
terrain.name = 'ridge-terrain';
scene.add(terrain);

const water = new Mesh(new PlaneGeometry(58, 45), new StandardMaterial({
  color: new Color().setHex(0x263c4b),
  metalness: 0.2,
  roughness: 0.12,
  clearcoat: 1,
  clearcoatRoughness: 0.08,
}));
water.name = 'mountain-lake';
water.position.set(-20, -2.1, 26);
scene.add(water);

const stats = new Stats();
if (stats.dom) {
  stats.dom.style.cssText = 'position:fixed;top:18px;right:18px';
  document.body.append(stats.dom);
}

const sunInput = document.querySelector<HTMLInputElement>('#sun')!;
const fogInput = document.querySelector<HTMLInputElement>('#fog')!;
const volumetricInput = document.querySelector<HTMLInputElement>('#volumetric')!;
const orbitInput = document.querySelector<HTMLInputElement>('#orbit')!;
sunInput.addEventListener('input', updateAtmosphere);
fogInput.addEventListener('input', updateAtmosphere);
volumetricInput.addEventListener('input', () => { renderer.volumetricFog = volumetricInput.checked; });
orbitInput.addEventListener('input', () => { controls.autoRotate = orbitInput.checked; });

function updateAtmosphere(): void {
  const height = Number(sunInput.value);
  const direction = new Vector3(0.68, height, -0.55).normalize();
  scene.sky!.sunDirection.copy(direction);
  sun.position.copy(direction).multiplyScalar(130);
  scene.fog!.density = Number(fogInput.value);
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
updateAtmosphere();

function animate(): void {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  stats.update();
}
animate();
