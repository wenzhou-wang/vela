import type { Scene } from '../core/Scene';
import type { Camera } from '../core/Camera';
import type { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { Light } from '../lights/Light';
import { AmbientLight } from '../lights/AmbientLight';
import { StandardMaterial } from '../materials/StandardMaterial';
import { ShaderMaterial } from '../materials/ShaderMaterial';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Frustum } from '../math/Frustum';
import { Sphere } from '../math/Sphere';
import { Matrix4 } from '../math/Matrix4';
import { Vector3 } from '../math/Vector3';
import { IrradianceProbeGrid } from '../core/IrradianceProbeGrid';
import { PerspectiveCamera } from '../core/PerspectiveCamera';

/** One finding from `renderer.diagnose()`: machine-matchable and human-fixable. */
export interface Diagnostic {
  severity: 'error' | 'warning';
  /** Stable identifier (e.g. 'camera-misses-scene') for programmatic handling. */
  code: string;
  /** What is wrong, naming the offending object. */
  message: string;
  /** The one-line fix. */
  fix: string;
}

/** Renderer state snapshot the checks need (kept GPU-free for offline testing). */
export interface DiagnoseState {
  postProcessing: boolean;
  sampleCount: number;
  oit: boolean;
  ssao: boolean;
  ssr: boolean;
  taa: boolean;
  shadows: boolean;
  shadowCascades?: number;
  volumetricFog?: boolean;
  motionBlur?: boolean;
  canvasWidth: number;
  canvasHeight: number;
}

const label = (o: Object3D): string => (o.name ? `"${o.name}" (${o.type})` : o.type);

/**
 * Triage why a scene "renders but looks wrong" — especially the classic black
 * screen. Pure scene-graph math: runs without a GPU. Returns an empty array
 * when nothing looks suspicious.
 */
export function diagnoseScene(scene: Scene, camera: Camera, state: DiagnoseState): Diagnostic[] {
  const out: Diagnostic[] = [];
  scene.updateMatrixWorld();
  camera.updateMatrixWorld();

  // ---- Gather scene contents + transform sanity --------------------------
  const meshes: Mesh[] = [];
  let litIntensity = 0;
  let hasIrradianceGrid = false;
  scene.traverseVisible((o: Object3D) => {
    if (o instanceof IrradianceProbeGrid) {
      if (o.coefficients) hasIrradianceGrid = true;
      else out.push({
        severity: 'warning', code: 'unbaked-irradiance-grid',
        message: `Irradiance grid ${label(o)} has no baked coefficients.`,
        fix: 'Call await renderer.bakeIrradianceProbes(grid, scene), then render again.',
      });
    }
    if (o instanceof Mesh) meshes.push(o);
    if (o instanceof Light) {
      const lum = o.color.r + o.color.g + o.color.b;
      litIntensity += o.intensity * lum;
      if (o.intensity === 0) {
        out.push({
          severity: 'warning', code: 'zero-intensity-light',
          message: `Light ${label(o)} has intensity 0 — it contributes nothing.`,
          fix: 'Set light.intensity > 0 or remove the light.',
        });
      } else if (lum === 0) {
        out.push({
          severity: 'warning', code: 'black-light',
          message: `Light ${label(o)} has a black color — it contributes nothing.`,
          fix: 'Set light.color to a non-black color.',
        });
      }
      if (!(o instanceof AmbientLight) && (o as { castShadow?: boolean }).castShadow && !state.shadows) {
        out.push({
          severity: 'warning', code: 'shadows-disabled',
          message: `Light ${label(o)} has castShadow = true but renderer.shadows is off.`,
          fix: 'Set renderer.shadows = true.',
        });
      }
    }
    const s = o.scale;
    if (s.x === 0 || s.y === 0 || s.z === 0) {
      out.push({
        severity: 'error', code: 'zero-scale',
        message: `${label(o)} has a zero scale component (${s.x}, ${s.y}, ${s.z}) — it and its children collapse to nothing.`,
        fix: 'Set all object.scale components to non-zero values.',
      });
    }
    for (const e of o.matrixWorld.elements) {
      if (Number.isNaN(e)) {
        out.push({
          severity: 'error', code: 'nan-transform',
          message: `${label(o)} has NaN in its world matrix — it will not render.`,
          fix: 'Check position/rotation/scale for NaN (often a divide-by-zero upstream).',
        });
        break;
      }
    }
  });

  if (meshes.length === 0) {
    out.push({
      severity: 'error', code: 'empty-scene',
      message: 'The scene contains no visible meshes.',
      fix: 'Add a Mesh to the scene (and check object.visible flags).',
    });
  }

  // ---- Lighting ------------------------------------------------------------
  const ambient = scene.ambientIntensity *
    (scene.ambientColor.r + scene.ambientColor.g + scene.ambientColor.b);
  const hasEnvLight = !!scene.environment || !!scene.sky;
  if (meshes.length > 0 && litIntensity === 0 && ambient === 0 && !hasEnvLight && !hasIrradianceGrid) {
    out.push({
      severity: 'error', code: 'no-lights',
      message: 'No lights, no ambient, and no environment/sky — PBR surfaces will render black (only emissive shows).',
      fix: 'Add a DirectionalLight, set scene.ambientIntensity > 0, or set scene.environment / scene.sky.',
    });
  }

  // ---- Materials -----------------------------------------------------------
  let scaleSamples: number[] = [];
  for (const mesh of meshes) {
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of materials) {
      if (!mat) {
        out.push({
          severity: 'error', code: 'missing-material',
          message: `Mesh ${label(mesh)} has no material.`,
          fix: 'Assign a StandardMaterial or ShaderMaterial.',
        });
      } else if (!(mat instanceof StandardMaterial) && !(mat instanceof ShaderMaterial) && !(mat instanceof LineBasicMaterial)) {
        out.push({
          severity: 'error', code: 'unsupported-material',
          message: `Mesh ${label(mesh)} uses material type "${mat.type}", which the renderer cannot draw — it is silently skipped.`,
          fix: 'Use StandardMaterial, ShaderMaterial, or LineBasicMaterial.',
        });
      } else if (mat.transparent && mat.opacity >= 1 && mat instanceof StandardMaterial && (!mat.map || mat.alphaTest === 0)) {
        out.push({
          severity: 'warning', code: 'transparent-opaque',
          message: `Mesh ${label(mesh)}'s material is transparent but fully opaque (opacity 1, no alpha source) — it pays blending/sorting cost for nothing.`,
          fix: 'Set material.opacity < 1 (or an alpha-carrying map), or set transparent = false.',
        });
      }
    }
    const geo = mesh.geometry;
    if (!geo.boundingSphere) geo.computeBoundingSphere();
    if (geo.boundingSphere && !geo.boundingSphere.isEmpty()) {
      const r = _sphere.copy(geo.boundingSphere).applyMatrix4(mesh.matrixWorld).radius;
      if (r > 0) scaleSamples.push(r);
    }
  }

  // ---- Scale outliers --------------------------------------------------------
  if (scaleSamples.length > 1) {
    const sorted = [...scaleSamples].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    for (const mesh of meshes) {
      const sphere = mesh.geometry.boundingSphere;
      if (!sphere || sphere.isEmpty()) continue;
      const r = _sphere.copy(sphere).applyMatrix4(mesh.matrixWorld).radius;
      if (r > median * 1000) {
        out.push({
          severity: 'warning', code: 'scale-outlier',
          message: `Mesh ${label(mesh)} is ${(r / median).toFixed(0)}× larger than the scene median — likely a unit mismatch (m vs cm).`,
          fix: 'Scale the object (or the rest of the scene) so sizes are comparable.',
        });
      } else if (r < median / 1000) {
        out.push({
          severity: 'warning', code: 'scale-outlier',
          message: `Mesh ${label(mesh)} is ${(median / r).toFixed(0)}× smaller than the scene median — likely a unit mismatch (m vs cm).`,
          fix: 'Scale the object (or the rest of the scene) so sizes are comparable.',
        });
      }
    }
  }

  // ---- Camera ----------------------------------------------------------------
  const cam = camera as unknown as { near?: number; far?: number; aspect?: number; fov?: number };
  if (cam.near !== undefined && cam.far !== undefined) {
    if (cam.near <= 0) {
      out.push({
        severity: 'error', code: 'camera-near-invalid',
        message: `camera.near is ${cam.near}; perspective projection requires near > 0.`,
        fix: 'Set camera.near to a small positive value (e.g. 0.1) and call updateProjectionMatrix().',
      });
    }
    if (cam.near >= cam.far) {
      out.push({
        severity: 'error', code: 'camera-near-far',
        message: `camera.near (${cam.near}) >= camera.far (${cam.far}) — nothing can be inside the frustum.`,
        fix: 'Set camera.far well above camera.near and call updateProjectionMatrix().',
      });
    } else if (cam.far / Math.max(cam.near, 1e-6) > 1e6) {
      out.push({
        severity: 'warning', code: 'depth-precision',
        message: `camera.far / camera.near = ${(cam.far / cam.near).toExponential(1)} — expect z-fighting from depth precision loss.`,
        fix: 'Raise camera.near (most effective) or lower camera.far.',
      });
    }
  }
  if (cam.aspect !== undefined && state.canvasHeight > 0) {
    const canvasAspect = state.canvasWidth / state.canvasHeight;
    if (Math.abs(cam.aspect - canvasAspect) / canvasAspect > 0.05) {
      out.push({
        severity: 'warning', code: 'aspect-mismatch',
        message: `camera.aspect (${cam.aspect.toFixed(3)}) differs from the canvas aspect (${canvasAspect.toFixed(3)}) — the image will look stretched.`,
        fix: 'Set camera.aspect = width / height and call updateProjectionMatrix() on resize.',
      });
    }
  }

  // ---- Does the camera actually see the scene? -------------------------------
  if (meshes.length > 0) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const mesh of meshes) {
      const s = mesh.geometry.boundingSphere;
      if (!s || s.isEmpty()) continue;
      _sphere.copy(s).applyMatrix4(mesh.matrixWorld);
      const c = _sphere.center, r = _sphere.radius;
      minX = Math.min(minX, c.x - r); maxX = Math.max(maxX, c.x + r);
      minY = Math.min(minY, c.y - r); maxY = Math.max(maxY, c.y + r);
      minZ = Math.min(minZ, c.z - r); maxZ = Math.max(maxZ, c.z + r);
    }
    if (minX <= maxX) {
      _center.set((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
      const radius = 0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
      _bounds.center.copy(_center);
      _bounds.radius = radius;
      _viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      _frustum.setFromProjectionMatrix(_viewProj);
      camera.getWorldPosition(_camPos);
      const dist = _camPos.distanceTo(_center);
      // Facing the scene but it lies beyond camera.far? Blame far, not aim.
      _toCenter.copy(_center).sub(_camPos).normalize();
      _viewDir.set(-camera.matrixWorld.elements[8], -camera.matrixWorld.elements[9], -camera.matrixWorld.elements[10]);
      const facing = _toCenter.dot(_viewDir) > 0.5;
      if (facing && cam.far !== undefined && dist - radius > cam.far) {
        out.push({
          severity: 'error', code: 'far-too-small',
          message: `The scene starts ${(dist - radius).toFixed(1)} units away but camera.far is ${cam.far} — everything is clipped.`,
          fix: `Set camera.far above ${Math.ceil(dist + radius)} and call updateProjectionMatrix().`,
        });
      } else if (!_frustum.intersectsSphere(_bounds)) {
        const d = Math.max(radius * 2.5, 1);
        const px = (_center.x + d * 0.55).toFixed(1);
        const py = (_center.y + d * 0.4).toFixed(1);
        const pz = (_center.z + d * 0.75).toFixed(1);
        out.push({
          severity: 'error', code: 'camera-misses-scene',
          message: `The camera frustum does not intersect the scene bounds ` +
            `(center ${fmt(_center)}, radius ${radius.toFixed(1)}; camera at ${fmt(_camPos)}, distance ${dist.toFixed(1)}).`,
          fix: `Point the camera at the scene, e.g. camera.position.set(${px}, ${py}, ${pz}); camera.lookAt(new Vector3(${fmt(_center)})).`,
        });
      }
    }
  }

  // ---- Renderer flag prerequisites -------------------------------------------
  const needsPost: Array<[string, boolean]> = [
    ['oit', state.oit],
    ['ssao', state.ssao],
    ['ssr', state.ssr],
    ['taa', state.taa],
    ['motionBlur', state.motionBlur ?? false],
  ];
  for (const [flag, enabled] of needsPost) {
    if (enabled && !state.postProcessing) {
      out.push({
        severity: 'warning', code: `${flag}-needs-post`,
        message: `renderer.${flag} is enabled but renderer.postProcessing is off — ${flag} is silently inactive.`,
        fix: 'Set renderer.postProcessing = true.',
      });
    }
    if (enabled && (flag === 'ssao' || flag === 'ssr' || flag === 'taa' || flag === 'motionBlur') && state.sampleCount !== 1) {
      out.push({
        severity: 'warning', code: `${flag}-needs-samplecount-1`,
        message: `renderer.${flag} requires sampleCount 1, but the renderer uses ${state.sampleCount}× MSAA — ${flag} is silently inactive.`,
        fix: 'Create the renderer with { sampleCount: 1 }.',
      });
    }
  }
  if (scene.skybox && !scene.environment && !scene.sky) {
    out.push({
      severity: 'warning', code: 'skybox-no-environment',
      message: 'scene.skybox is true but neither scene.environment nor scene.sky is set — there is nothing to draw.',
      fix: 'Set scene.environment (an equirect texture) or scene.sky ({ sunDirection }).',
    });
  }
  const shadowCascades = state.shadowCascades ?? 1;
  if (state.shadows && shadowCascades > 1 && !(camera instanceof PerspectiveCamera)) {
    out.push({
      severity: 'warning', code: 'csm-needs-perspective-camera',
      message: `renderer.shadowCascades is ${shadowCascades}, but cascades require a PerspectiveCamera; one shadow map is used.`,
      fix: 'Use a PerspectiveCamera or set renderer.shadowCascades = 1.',
    });
  }
  if (state.volumetricFog && !scene.fog) {
    out.push({
      severity: 'warning', code: 'volumetric-fog-needs-fog',
      message: 'renderer.volumetricFog is enabled but scene.fog is null — the froxel pass is inactive.',
      fix: 'Set scene.fog or disable renderer.volumetricFog.',
    });
  }
  if (scene.fog && scene.fog.density === undefined) {
    const near = scene.fog.near ?? 1;
    const far = scene.fog.far ?? 100;
    if (far <= near) {
      out.push({
        severity: 'warning', code: 'fog-range-invalid',
        message: `scene.fog has far (${far}) <= near (${near}) — everything beyond ${near} is fully fogged.`,
        fix: 'Set fog.far above fog.near.',
      });
    }
  }

  return out;
}

const _sphere = new Sphere();
const _bounds = new Sphere();
const _frustum = new Frustum();
const _viewProj = new Matrix4();
const _center = new Vector3();
const _camPos = new Vector3();
const _toCenter = new Vector3();
const _viewDir = new Vector3();
const fmt = (v: Vector3) => `${v.x.toFixed(1)}, ${v.y.toFixed(1)}, ${v.z.toFixed(1)}`;
