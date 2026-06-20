import { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { LineSegments } from '../core/LineSegments';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute, type TypedArray } from '../core/BufferAttribute';
import { Material } from '../materials/Material';
import { StandardMaterial } from '../materials/StandardMaterial';
import { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { Light } from '../lights/Light';
import { AmbientLight } from '../lights/AmbientLight';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { Color } from '../math/Color';
import { IrradianceProbeGrid } from '../core/IrradianceProbeGrid';
import { Vector3 } from '../math/Vector3';

type Json = Record<string, unknown>;

/**
 * vela's native scene format: a compact, lossless-enough JSON round-trip for the
 * scene graph — `Object3D`/`Mesh`/`LineSegments`/lights/irradiance grids, their transforms,
 * `BufferGeometry` attributes, and `StandardMaterial`/`LineBasicMaterial`.
 * Geometries and materials are de-duplicated into shared tables. (Textures,
 * skinning, morphs, and animation are not serialized.)
 */
export class SceneSerializer {
  static readonly FORMAT = 'vela-scene';
  static readonly VERSION = 1;

  /** Serialize a subtree to a plain JSON-safe object. */
  static serialize(root: Object3D): Json {
    const geometries: Json = {};
    const materials: Json = {};

    const serializeGeometry = (geo: BufferGeometry): string => {
      if (!geometries[geo.id]) geometries[geo.id] = geometryToJSON(geo);
      return geo.id;
    };
    const serializeMaterial = (mat: Material): string => {
      if (!materials[mat.id]) materials[mat.id] = materialToJSON(mat);
      return mat.id;
    };

    const node = (object: Object3D): Json => {
      const out: Json = { type: object.type, name: object.name };
      const p = object.position, q = object.quaternion, s = object.scale;
      out.position = [p.x, p.y, p.z];
      out.quaternion = [q.x, q.y, q.z, q.w];
      out.scale = [s.x, s.y, s.z];
      out.visible = object.visible;

      if (object instanceof IrradianceProbeGrid) {
        out.dimensions = object.dimensions;
        out.spacing = [object.spacing.x, object.spacing.y, object.spacing.z];
        if (object.coefficients) out.coefficients = Array.from(object.coefficients);
      } else if (object instanceof Mesh) {
        out.geometry = serializeGeometry(object.geometry);
        const mat = Array.isArray(object.material) ? object.material[0] : object.material;
        out.material = serializeMaterial(mat);
      } else if (object instanceof Light) {
        lightToJSON(object, out);
      }

      if (object.children.length) out.children = object.children.map(node);
      return out;
    };

    return {
      metadata: { format: SceneSerializer.FORMAT, version: SceneSerializer.VERSION },
      geometries,
      materials,
      object: node(root),
    };
  }

  /** Rebuild a subtree from {@link serialize} output. */
  static deserialize(json: Json): Object3D {
    const geometries = new Map<string, BufferGeometry>();
    for (const [id, g] of Object.entries(json.geometries as Json)) {
      geometries.set(id, geometryFromJSON(g as Json));
    }
    const materials = new Map<string, Material>();
    for (const [id, m] of Object.entries(json.materials as Json)) {
      materials.set(id, materialFromJSON(m as Json));
    }

    const build = (data: Json): Object3D => {
      const object = createObject(data, geometries, materials);
      object.name = (data.name as string) ?? '';
      const p = data.position as number[];
      const q = data.quaternion as number[];
      const s = data.scale as number[];
      if (p) object.position.set(p[0], p[1], p[2]);
      if (q) object.quaternion.set(q[0], q[1], q[2], q[3]);
      if (s) object.scale.set(s[0], s[1], s[2]);
      if (data.visible !== undefined) object.visible = data.visible as boolean;
      for (const child of (data.children as Json[]) ?? []) object.add(build(child));
      return object;
    };

    return build(json.object as Json);
  }
}

// ---- geometry ----

function geometryToJSON(geo: BufferGeometry): Json {
  const attributes: Json = {};
  for (const [name, attr] of Object.entries(geo.attributes)) {
    attributes[name] = { array: Array.from(attr.array), itemSize: attr.itemSize, normalized: attr.normalized };
  }
  const out: Json = { name: geo.name, attributes };
  if (geo.index) {
    out.index = { array: Array.from(geo.index.array), type: geo.index.array.constructor.name };
  }
  return out;
}

function geometryFromJSON(data: Json): BufferGeometry {
  const geo = new BufferGeometry();
  geo.name = (data.name as string) ?? '';
  for (const [name, a] of Object.entries(data.attributes as Json)) {
    const attr = a as { array: number[]; itemSize: number; normalized?: boolean };
    geo.setAttribute(name, new BufferAttribute(new Float32Array(attr.array), attr.itemSize, attr.normalized));
  }
  if (data.index) {
    const idx = data.index as { array: number[]; type: string };
    const Ctor = idx.type === 'Uint16Array' ? Uint16Array : Uint32Array;
    geo.setIndex(new BufferAttribute(new Ctor(idx.array) as TypedArray, 1));
  }
  return geo;
}

// ---- material ----

const rgb = (c: Color): number[] => [c.r, c.g, c.b];

function materialToJSON(material: Material): Json {
  const out: Json = {
    type: material.type, name: material.name,
    side: material.side, transparent: material.transparent, opacity: material.opacity,
    depthTest: material.depthTest, depthWrite: material.depthWrite, alphaTest: material.alphaTest,
  };
  if (material instanceof StandardMaterial) {
    out.color = rgb(material.color);
    out.metalness = material.metalness;
    out.roughness = material.roughness;
    out.emissive = rgb(material.emissive);
    out.emissiveIntensity = material.emissiveIntensity;
    out.normalScale = material.normalScale;
    out.occlusionStrength = material.occlusionStrength;
    out.clearcoat = material.clearcoat;
    out.clearcoatRoughness = material.clearcoatRoughness;
    out.ior = material.ior;
    out.specularIntensity = material.specularIntensity;
    out.specularColor = rgb(material.specularColor);
    out.sheenColor = rgb(material.sheenColor);
    out.sheenRoughness = material.sheenRoughness;
    out.parallaxScale = material.parallaxScale;
    out.parallaxMinLayers = material.parallaxMinLayers;
    out.parallaxMaxLayers = material.parallaxMaxLayers;
  } else if (material instanceof LineBasicMaterial) {
    out.color = rgb(material.color);
    out.vertexColors = material.vertexColors;
  }
  return out;
}

function materialFromJSON(data: Json): Material {
  let material: Material;
  if (data.type === 'LineBasicMaterial') {
    const m = new LineBasicMaterial();
    m.color.setRGB(...(data.color as [number, number, number]));
    m.vertexColors = (data.vertexColors as boolean) ?? false;
    material = m;
  } else {
    const m = new StandardMaterial();
    const target = m as unknown as Record<string, unknown>;
    const set = (k: string, v: unknown) => { if (v !== undefined) target[k] = v; };
    if (data.color) m.color.setRGB(...(data.color as [number, number, number]));
    if (data.emissive) m.emissive.setRGB(...(data.emissive as [number, number, number]));
    if (data.specularColor) m.specularColor.setRGB(...(data.specularColor as [number, number, number]));
    if (data.sheenColor) m.sheenColor.setRGB(...(data.sheenColor as [number, number, number]));
    set('metalness', data.metalness);
    set('roughness', data.roughness);
    set('emissiveIntensity', data.emissiveIntensity);
    set('normalScale', data.normalScale);
    set('occlusionStrength', data.occlusionStrength);
    set('parallaxScale', data.parallaxScale);
    set('parallaxMinLayers', data.parallaxMinLayers);
    set('parallaxMaxLayers', data.parallaxMaxLayers);
    set('clearcoat', data.clearcoat);
    set('clearcoatRoughness', data.clearcoatRoughness);
    set('ior', data.ior);
    set('specularIntensity', data.specularIntensity);
    set('sheenRoughness', data.sheenRoughness);
    material = m;
  }
  material.name = (data.name as string) ?? '';
  if (data.side !== undefined) material.side = data.side as Material['side'];
  if (data.transparent !== undefined) material.transparent = data.transparent as boolean;
  if (data.opacity !== undefined) material.opacity = data.opacity as number;
  if (data.depthTest !== undefined) material.depthTest = data.depthTest as boolean;
  if (data.depthWrite !== undefined) material.depthWrite = data.depthWrite as boolean;
  if (data.alphaTest !== undefined) material.alphaTest = data.alphaTest as number;
  return material;
}

// ---- lights ----

function lightToJSON(light: Light, out: Json): void {
  out.color = rgb(light.color);
  out.intensity = light.intensity;
  if (light instanceof PointLight) {
    out.distance = light.distance;
    out.decay = light.decay;
  } else if (light instanceof DirectionalLight) {
    const t = light.target.position;
    out.target = [t.x, t.y, t.z];
  }
}

function createObject(
  data: Json,
  geometries: Map<string, BufferGeometry>,
  materials: Map<string, Material>,
): Object3D {
  switch (data.type) {
    case 'IrradianceProbeGrid': {
      const spacing = (data.spacing as [number, number, number]) ?? [2, 2, 2];
      const grid = new IrradianceProbeGrid({
        dimensions: data.dimensions as [number, number, number],
        spacing: new Vector3(spacing[0], spacing[1], spacing[2]),
      });
      if (data.coefficients) grid.setCoefficients(new Float32Array(data.coefficients as number[]));
      return grid;
    }
    case 'Mesh':
    case 'SkinnedMesh':
    case 'InstancedMesh': {
      const geo = geometries.get(data.geometry as string)!;
      const mat = materials.get(data.material as string)!;
      return new Mesh(geo, mat);
    }
    case 'LineSegments':
    case 'GridHelper':
    case 'AxesHelper':
    case 'Box3Helper': {
      const geo = geometries.get(data.geometry as string)!;
      const mat = materials.get(data.material as string) as LineBasicMaterial;
      return new LineSegments(geo, mat);
    }
    case 'AmbientLight':
      return applyLight(new AmbientLight(), data);
    case 'DirectionalLight': {
      const l = applyLight(new DirectionalLight(), data) as DirectionalLight;
      const t = data.target as number[] | undefined;
      if (t) l.target.position.set(t[0], t[1], t[2]);
      return l;
    }
    case 'PointLight': {
      const l = applyLight(new PointLight(), data) as PointLight;
      if (data.distance !== undefined) l.distance = data.distance as number;
      if (data.decay !== undefined) l.decay = data.decay as number;
      return l;
    }
    default:
      return new Object3D();
  }
}

function applyLight(light: Light, data: Json): Light {
  if (data.color) light.color.setRGB(...(data.color as [number, number, number]));
  if (data.intensity !== undefined) light.intensity = data.intensity as number;
  return light;
}
