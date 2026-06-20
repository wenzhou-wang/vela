import type { Scene } from './Scene';
import type { Camera } from './Camera';
import type { Object3D } from './Object3D';
import { Mesh } from './Mesh';
import { InstancedMesh } from './InstancedMesh';
import { SkinnedMesh } from './SkinnedMesh';
import { ParticleSystem } from './ParticleSystem';
import { Sprite } from './Sprite';
import { TextMesh } from './TextMesh';
import { ReflectionProbe } from './ReflectionProbe';
import { IrradianceProbeGrid } from './IrradianceProbeGrid';
import { Decal } from './Decal';
import { Light } from '../lights/Light';
import type { Material } from '../materials/Material';
import { Frustum } from '../math/Frustum';
import { Sphere } from '../math/Sphere';
import { Matrix4 } from '../math/Matrix4';

export interface SceneDescription {
  counts: {
    nodes: number;
    meshes: number;
    instancedMeshes: number;
    skinnedMeshes: number;
    particleSystems: number;
    sprites: number;
    texts: number;
    reflectionProbes: number;
    irradianceProbeGrids: number;
    decals: number;
    lights: Record<string, number>;
  };
  /** World-space bounds of all meshes (null when empty). */
  bounds: { center: [number, number, number]; size: [number, number, number]; radius: number } | null;
  /** Unique materials with how many meshes use each. */
  materials: Array<{ type: string; name: string; users: number; transparent: boolean }>;
  environment: 'texture' | 'procedural-sky' | 'none';
  skybox: boolean;
  fog: 'linear' | 'exp2' | 'none';
  /**
   * Named nodes (capped at 100) with world position — and, when a camera is
   * provided, whether each mesh is inside its frustum.
   */
  named: Array<{ name: string; type: string; position: [number, number, number]; inFrustum?: boolean }>;
  /** Present when a camera is given: how much of the scene it can see. */
  visibility?: { meshesTested: number; inFrustum: number; outOfFrustum: number };
}

const _sphere = new Sphere();
const _frustum = new Frustum();
const _viewProj = new Matrix4();

/**
 * Summarize a scene as plain JSON an agent can read: node/light/material
 * inventories, world bounds, environment/fog state, named nodes, and (with a
 * camera) per-mesh frustum visibility. Pure scene-graph math — no GPU.
 */
export function describeScene(scene: Scene, camera?: Camera): SceneDescription {
  scene.updateMatrixWorld();
  let frustumReady = false;
  if (camera) {
    camera.updateMatrixWorld();
    _viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_viewProj);
    frustumReady = true;
  }

  const counts = {
    nodes: 0, meshes: 0, instancedMeshes: 0, skinnedMeshes: 0,
    particleSystems: 0, sprites: 0, texts: 0, reflectionProbes: 0, irradianceProbeGrids: 0, decals: 0,
    lights: {} as Record<string, number>,
  };
  const named: SceneDescription['named'] = [];
  const materialUsers = new Map<Material, number>();
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  let meshesTested = 0;
  let inFrustum = 0;

  scene.traverseVisible((o: Object3D) => {
    counts.nodes++;
    let meshInFrustum: boolean | undefined;
    if (o instanceof Light) {
      counts.lights[o.type] = (counts.lights[o.type] ?? 0) + 1;
    } else if (o instanceof ParticleSystem) {
      counts.particleSystems++;
    } else if (o instanceof Sprite) {
      counts.sprites++;
    } else if (o instanceof TextMesh) {
      counts.texts++;
    } else if (o instanceof ReflectionProbe) {
      counts.reflectionProbes++;
    } else if (o instanceof IrradianceProbeGrid) {
      counts.irradianceProbeGrids++;
    } else if (o instanceof Decal) {
      counts.decals++;
    } else if (o instanceof Mesh) {
      counts.meshes++;
      if (o instanceof InstancedMesh) counts.instancedMeshes++;
      if (o instanceof SkinnedMesh) counts.skinnedMeshes++;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) if (m) materialUsers.set(m, (materialUsers.get(m) ?? 0) + 1);
      const geo = o.geometry;
      if (!geo.boundingSphere) geo.computeBoundingSphere();
      if (geo.boundingSphere && !geo.boundingSphere.isEmpty()) {
        _sphere.copy(geo.boundingSphere).applyMatrix4(o.matrixWorld);
        const c = _sphere.center, r = _sphere.radius;
        minX = Math.min(minX, c.x - r); maxX = Math.max(maxX, c.x + r);
        minY = Math.min(minY, c.y - r); maxY = Math.max(maxY, c.y + r);
        minZ = Math.min(minZ, c.z - r); maxZ = Math.max(maxZ, c.z + r);
        if (frustumReady) {
          meshesTested++;
          meshInFrustum = _frustum.intersectsSphere(_sphere);
          if (meshInFrustum) inFrustum++;
        }
      }
    }
    if (o.name && named.length < 100) {
      const e = o.matrixWorld.elements;
      const entry: SceneDescription['named'][number] = {
        name: o.name,
        type: o.type,
        position: [round(e[12]), round(e[13]), round(e[14])],
      };
      if (meshInFrustum !== undefined) entry.inFrustum = meshInFrustum;
      named.push(entry);
    }
  });

  const materials = [...materialUsers.entries()].map(([m, users]) => ({
    type: m.type,
    name: m.name,
    users,
    transparent: m.transparent,
  }));

  const description: SceneDescription = {
    counts,
    bounds: minX <= maxX
      ? {
          center: [round((minX + maxX) / 2), round((minY + maxY) / 2), round((minZ + maxZ) / 2)],
          size: [round(maxX - minX), round(maxY - minY), round(maxZ - minZ)],
          radius: round(0.5 * Math.hypot(maxX - minX, maxY - minY, maxZ - minZ)),
        }
      : null,
    materials,
    environment: scene.environment ? 'texture' : scene.sky ? 'procedural-sky' : 'none',
    skybox: scene.skybox,
    fog: scene.fog ? (scene.fog.density !== undefined ? 'exp2' : 'linear') : 'none',
    named,
  };
  if (frustumReady) {
    description.visibility = { meshesTested, inFrustum, outOfFrustum: meshesTested - inFrustum };
  }
  return description;
}

const round = (v: number) => Math.round(v * 100) / 100;
