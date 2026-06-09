import { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { SkinnedMesh } from '../core/SkinnedMesh';
import { StandardMaterial } from '../materials/StandardMaterial';
import type { Texture, WrapMode, FilterMode } from '../textures/Texture';
import type { AnimationClip } from '../animation/AnimationClip';

/** Result of an export: the glTF JSON plus the packed binary (a `.glb`). */
export interface GLTFExportResult {
  json: Record<string, unknown>;
  /** A self-contained binary glTF (`.glb`) buffer. */
  glb: ArrayBuffer;
}

const FLOAT = 5126;
const UNSIGNED_INT = 5125;
const UNSIGNED_SHORT = 5123;
const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;
const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

/**
 * Exports a scene-graph subtree to glTF 2.0 (binary `.glb`). Covers the
 * round-trippable core: node hierarchy + TRS, mesh geometry (position / normal /
 * uv / color / indices), `StandardMaterial` PBR factors + textures, clearcoat /
 * ior / specular / sheen extensions, morph targets, skinning, and animations.
 * All public methods are async to support texture encoding.
 */
export class GLTFExporter {
  private json!: {
    asset: { version: string; generator: string };
    scene: number;
    scenes: { nodes: number[] }[];
    nodes: Record<string, unknown>[];
    meshes: Record<string, unknown>[];
    materials: Record<string, unknown>[];
    accessors: Record<string, unknown>[];
    bufferViews: Record<string, unknown>[];
    buffers: { byteLength: number }[];
    animations?: Record<string, unknown>[];
    images?: Record<string, unknown>[];
    samplers?: Record<string, unknown>[];
    textures?: Record<string, unknown>[];
  };
  private chunks!: Uint8Array[];
  private byteLength!: number;
  private materialIndex!: Map<StandardMaterial, number>;
  private meshIndex!: Map<string, number>;
  private nodeMap!: Map<Object3D, number>;
  private skinnedMeshes!: SkinnedMesh[];
  private textureIndex!: Map<string, number>;
  private samplerIndex!: Map<string, number>;

  async parse(root: Object3D, animations: AnimationClip[] = []): Promise<GLTFExportResult> {
    const binary = await this.assemble(root, animations);
    this.json.buffers.push({ byteLength: binary.byteLength });
    return { json: this.json, glb: this.buildGLB(this.json, binary) };
  }

  /** Convenience: export just the `.glb` ArrayBuffer. */
  async parseGLB(root: Object3D, animations: AnimationClip[] = []): Promise<ArrayBuffer> {
    return (await this.parse(root, animations)).glb;
  }

  /**
   * Export to a standalone `.gltf` JSON object with the binary embedded as a
   * base64 data-URI buffer (re-loadable directly by `GLTFLoader`).
   */
  async parseGLTF(root: Object3D, animations: AnimationClip[] = []): Promise<Record<string, unknown>> {
    const binary = await this.assemble(root, animations);
    this.json.buffers.push({
      byteLength: binary.byteLength,
      uri: 'data:application/octet-stream;base64,' + base64Encode(binary),
    } as { byteLength: number });
    return this.json;
  }

  /** Build the glTF JSON + packed binary; shared by every output format. */
  private async assemble(root: Object3D, animations: AnimationClip[]): Promise<Uint8Array> {
    this.json = {
      asset: { version: '2.0', generator: 'vela GLTFExporter' },
      scene: 0,
      scenes: [{ nodes: [] }],
      nodes: [],
      meshes: [],
      materials: [],
      accessors: [],
      bufferViews: [],
      buffers: [],
    };
    this.chunks = [];
    this.byteLength = 0;
    this.materialIndex = new Map();
    this.meshIndex = new Map();
    this.nodeMap = new Map();
    this.skinnedMeshes = [];
    this.textureIndex = new Map();
    this.samplerIndex = new Map();

    root.updateMatrixWorld(true);
    const topLevel = root instanceof Mesh ? [root] : root.children;
    const nodeIndices: number[] = [];
    for (const child of topLevel) nodeIndices.push(await this.addNode(child));
    this.json.scenes[0].nodes = nodeIndices;

    this.addSkins();
    if (animations.length) this.addAnimations(animations);

    if (this.json.materials.length === 0) delete (this.json as Record<string, unknown>).materials;
    return this.concatChunks();
  }

  private async addNode(object: Object3D): Promise<number> {
    const node: Record<string, unknown> = {};
    if (object.name) node.name = object.name;

    const p = object.position;
    if (p.x !== 0 || p.y !== 0 || p.z !== 0) node.translation = [p.x, p.y, p.z];
    const q = object.quaternion;
    if (q.x !== 0 || q.y !== 0 || q.z !== 0 || q.w !== 1) node.rotation = [q.x, q.y, q.z, q.w];
    const s = object.scale;
    if (s.x !== 1 || s.y !== 1 || s.z !== 1) node.scale = [s.x, s.y, s.z];

    if (object instanceof Mesh) {
      const material = Array.isArray(object.material) ? object.material[0] : object.material;
      node.mesh = await this.addMesh(object, material instanceof StandardMaterial ? material : null);
      if (object instanceof SkinnedMesh) this.skinnedMeshes.push(object);
    }

    const index = this.json.nodes.push(node) - 1; // reserve index before recursing
    this.nodeMap.set(object, index);
    const children: number[] = [];
    for (const child of object.children) children.push(await this.addNode(child));
    if (children.length) node.children = children;
    return index;
  }

  private addSkins(): void {
    if (!this.skinnedMeshes.length) return;
    const skins: Record<string, unknown>[] = [];
    for (const mesh of this.skinnedMeshes) {
      const nodeIndex = this.nodeMap.get(mesh);
      const skeleton = mesh.skeleton;
      if (nodeIndex === undefined || !skeleton || skeleton.jointCount === 0) continue;

      const jointIndices: number[] = [];
      let allMapped = true;
      for (const joint of skeleton.joints) {
        const ji = this.nodeMap.get(joint);
        if (ji === undefined) { allMapped = false; break; }
        jointIndices.push(ji);
      }
      if (!allMapped) continue;

      const ibm = new Float32Array(skeleton.jointCount * 16);
      for (let i = 0; i < skeleton.boneInverses.length; i++) ibm.set(skeleton.boneInverses[i].elements, i * 16);
      const inverseBindMatrices = this.addAccessor(ibm, 16, FLOAT, undefined);

      const skinIndex = skins.push({ joints: jointIndices, inverseBindMatrices }) - 1;
      this.json.nodes[nodeIndex].skin = skinIndex;
    }
    if (skins.length) (this.json as Record<string, unknown>).skins = skins;
  }

  private addAnimations(clips: AnimationClip[]): void {
    const animations: Record<string, unknown>[] = [];
    for (const clip of clips) {
      const samplers: Record<string, unknown>[] = [];
      const channels: Record<string, unknown>[] = [];
      for (const track of clip.tracks) {
        const nodeIndex = this.nodeMap.get(track.target);
        if (nodeIndex === undefined) continue;
        const input = this.addAccessor(track.times as Float32Array, 1, FLOAT, undefined, true);
        const comps = track.path === 'rotation' ? 4 : track.path === 'weights' ? 1 : 3;
        const output = this.addAccessor(track.values as Float32Array, comps, FLOAT, undefined);
        const sampler = samplers.push({ input, output, interpolation: track.interpolation }) - 1;
        channels.push({ sampler, target: { node: nodeIndex, path: track.path } });
      }
      if (channels.length) {
        const anim: Record<string, unknown> = { samplers, channels };
        if (clip.name) anim.name = clip.name;
        animations.push(anim);
      }
    }
    if (animations.length) this.json.animations = animations;
  }

  private async addMesh(mesh: Mesh, material: StandardMaterial | null): Promise<number> {
    const geometry = mesh.geometry;
    const matIndex = material ? await this.addMaterial(material) : -1;
    const key = `${geometry.id}|${matIndex}|${mesh.morphTargetInfluences.join(',')}`;
    const cached = this.meshIndex.get(key);
    if (cached !== undefined) return cached;

    const attributes: Record<string, number> = {};
    const pos = geometry.attributes.position;
    attributes.POSITION = this.addAccessor(pos.array as Float32Array, pos.itemSize, FLOAT, ARRAY_BUFFER, true);
    const normal = geometry.attributes.normal;
    if (normal) attributes.NORMAL = this.addAccessor(normal.array as Float32Array, 3, FLOAT, ARRAY_BUFFER);
    const uv = geometry.attributes.uv;
    if (uv) attributes.TEXCOORD_0 = this.addAccessor(uv.array as Float32Array, 2, FLOAT, ARRAY_BUFFER);
    const color = geometry.attributes.color;
    if (color) attributes.COLOR_0 = this.addAccessor(color.array as Float32Array, color.itemSize, FLOAT, ARRAY_BUFFER);
    const joints = geometry.attributes.joints;
    const weights = geometry.attributes.weights;
    if (joints && weights) {
      const u16 = Uint16Array.from(joints.array as ArrayLike<number>);
      attributes.JOINTS_0 = this.addAccessor(u16, 4, UNSIGNED_SHORT, ARRAY_BUFFER);
      attributes.WEIGHTS_0 = this.addAccessor(weights.array as Float32Array, 4, FLOAT, ARRAY_BUFFER);
    }

    const primitive: Record<string, unknown> = { attributes };
    if (geometry.index) {
      const u32 = geometry.index.array instanceof Uint32Array
        ? (geometry.index.array as Uint32Array)
        : Uint32Array.from(geometry.index.array as ArrayLike<number>);
      primitive.indices = this.addAccessor(u32, 1, UNSIGNED_INT, ELEMENT_ARRAY_BUFFER);
    }
    if (matIndex >= 0) primitive.material = matIndex;

    const morphPos = geometry.morphAttributes.position;
    if (morphPos?.length) {
      const morphNrm = geometry.morphAttributes.normal;
      primitive.targets = morphPos.map((p, i) => {
        const target: Record<string, number> = {
          POSITION: this.addAccessor(p.array as Float32Array, 3, FLOAT, ARRAY_BUFFER, true),
        };
        if (morphNrm?.[i]) target.NORMAL = this.addAccessor(morphNrm[i].array as Float32Array, 3, FLOAT, ARRAY_BUFFER);
        return target;
      });
    }

    const meshDef: Record<string, unknown> = { primitives: [primitive] };
    if (geometry.name) meshDef.name = geometry.name;
    if (morphPos?.length) {
      meshDef.weights = mesh.morphTargetInfluences.slice();
      if (mesh.morphTargetDictionary) {
        const names: string[] = [];
        for (const [name, i] of Object.entries(mesh.morphTargetDictionary)) names[i] = name;
        if (names.length === morphPos.length) meshDef.extras = { targetNames: names };
      }
    }
    const index = this.json.meshes.push(meshDef) - 1;
    this.meshIndex.set(key, index);
    return index;
  }

  private async addMaterial(material: StandardMaterial): Promise<number> {
    const cached = this.materialIndex.get(material);
    if (cached !== undefined) return cached;

    const c = material.color;
    const pbr: Record<string, unknown> = {
      baseColorFactor: [c.r, c.g, c.b, material.opacity],
      metallicFactor: material.metalness,
      roughnessFactor: material.roughness,
    };

    const ti = async (tex: Texture | null) => await this.addTextureRef(tex);

    const baseColorTi = await ti(material.map);
    if (baseColorTi >= 0) pbr.baseColorTexture = { index: baseColorTi };

    const mrTi = await ti(material.metalnessRoughnessMap);
    if (mrTi >= 0) pbr.metallicRoughnessTexture = { index: mrTi };

    const def: Record<string, unknown> = { pbrMetallicRoughness: pbr };
    if (material.name) def.name = material.name;

    const normalTi = await ti(material.normalMap);
    if (normalTi >= 0) def.normalTexture = { index: normalTi, scale: material.normalScale };

    const occTi = await ti(material.occlusionMap);
    if (occTi >= 0) def.occlusionTexture = { index: occTi, strength: material.occlusionStrength };

    const emiTi = await ti(material.emissiveMap);
    if (emiTi >= 0) def.emissiveTexture = { index: emiTi };

    const e = material.emissive;
    if (e.r !== 0 || e.g !== 0 || e.b !== 0) def.emissiveFactor = [e.r, e.g, e.b];

    if (material.side === 'double') def.doubleSided = true;
    if (material.transparent) def.alphaMode = 'BLEND';
    else if (material.alphaTest > 0) { def.alphaMode = 'MASK'; def.alphaCutoff = material.alphaTest; }

    const extensions: Record<string, unknown> = {};
    if (material.emissiveIntensity !== 1) {
      extensions.KHR_materials_emissive_strength = { emissiveStrength: material.emissiveIntensity };
    }
    if (material.clearcoat > 0) {
      const ccTi = await ti(material.clearcoatMap);
      const ccrTi = await ti(material.clearcoatRoughnessMap);
      const ccExt: Record<string, unknown> = {
        clearcoatFactor: material.clearcoat,
        clearcoatRoughnessFactor: material.clearcoatRoughness,
      };
      if (ccTi >= 0) ccExt.clearcoatTexture = { index: ccTi };
      if (ccrTi >= 0) ccExt.clearcoatRoughnessTexture = { index: ccrTi };
      extensions.KHR_materials_clearcoat = ccExt;
    }
    if (material.ior !== 1.5) extensions.KHR_materials_ior = { ior: material.ior };
    if (material.specularIntensity !== 1 || material.specularColor.r !== 1 ||
        material.specularColor.g !== 1 || material.specularColor.b !== 1) {
      extensions.KHR_materials_specular = {
        specularFactor: material.specularIntensity,
        specularColorFactor: [material.specularColor.r, material.specularColor.g, material.specularColor.b],
      };
    }
    if (material.sheenColor.r !== 0 || material.sheenColor.g !== 0 || material.sheenColor.b !== 0) {
      extensions.KHR_materials_sheen = {
        sheenColorFactor: [material.sheenColor.r, material.sheenColor.g, material.sheenColor.b],
        sheenRoughnessFactor: material.sheenRoughness,
      };
    }
    if (material.transmission > 0) {
      extensions.KHR_materials_transmission = { transmissionFactor: material.transmission };
    }
    if (material.thickness > 0 || material.attenuationDistance > 0) {
      extensions.KHR_materials_volume = {
        thicknessFactor: material.thickness,
        attenuationDistance: material.attenuationDistance,
        attenuationColor: [material.attenuationColor.r, material.attenuationColor.g, material.attenuationColor.b],
      };
    }
    if (Object.keys(extensions).length) def.extensions = extensions;

    const index = this.json.materials.push(def) - 1;
    this.materialIndex.set(material, index);
    return index;
  }

  /**
   * Encode a Texture's source as PNG and append it to the binary buffer.
   * Returns the glTF texture index, or -1 if the source is absent or
   * OffscreenCanvas is unavailable.
   */
  private async addTextureRef(texture: Texture | null): Promise<number> {
    if (!texture?.source) return -1;
    const cached = this.textureIndex.get(texture.id);
    if (cached !== undefined) return cached;

    const pngBytes = await encodeSource(texture.source);
    if (!pngBytes.length) return -1;

    this.align4();
    const byteOffset = this.byteLength;
    this.chunks.push(pngBytes);
    this.byteLength += pngBytes.byteLength;

    const bvIndex = this.json.bufferViews.push({
      buffer: 0, byteOffset, byteLength: pngBytes.byteLength,
    }) - 1;

    const images = (this.json.images ??= []);
    const imageIndex = images.push({ mimeType: 'image/png', bufferView: bvIndex }) - 1;

    const samplerKey = `${texture.wrapS}|${texture.wrapT}|${texture.magFilter}|${texture.minFilter}`;
    let si = this.samplerIndex.get(samplerKey);
    if (si === undefined) {
      const samplers = (this.json.samplers ??= []);
      si = samplers.push({
        wrapS: glWrap(texture.wrapS),
        wrapT: glWrap(texture.wrapT),
        magFilter: glFilter(texture.magFilter),
        minFilter: glFilter(texture.minFilter),
      }) - 1;
      this.samplerIndex.set(samplerKey, si);
    }

    const textures = (this.json.textures ??= []);
    const textureIndex = textures.push({ source: imageIndex, sampler: si }) - 1;
    this.textureIndex.set(texture.id, textureIndex);
    return textureIndex;
  }

  /** Append a typed array as a bufferView + accessor; returns the accessor index. */
  private addAccessor(
    data: Float32Array | Uint32Array | Uint16Array,
    itemSize: number,
    componentType: number,
    target?: number,
    computeMinMax = false,
  ): number {
    this.align4();
    const byteOffset = this.byteLength;
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    this.chunks.push(bytes);
    this.byteLength += bytes.byteLength;

    const viewDef: Record<string, unknown> = { buffer: 0, byteOffset, byteLength: bytes.byteLength };
    if (target !== undefined) viewDef.target = target;
    const bufferView = this.json.bufferViews.push(viewDef) - 1;

    const TYPE = itemSize === 16 ? 'MAT4' : ['', 'SCALAR', 'VEC2', 'VEC3', 'VEC4'][itemSize];
    const accessor: Record<string, unknown> = {
      bufferView, componentType, count: data.length / itemSize, type: TYPE,
    };
    if (computeMinMax) {
      const min = new Array(itemSize).fill(Infinity);
      const max = new Array(itemSize).fill(-Infinity);
      for (let i = 0; i < data.length; i += itemSize) {
        for (let k = 0; k < itemSize; k++) {
          const v = data[i + k];
          if (v < min[k]) min[k] = v;
          if (v > max[k]) max[k] = v;
        }
      }
      accessor.min = min;
      accessor.max = max;
    }
    return this.json.accessors.push(accessor) - 1;
  }

  private align4(): void {
    const pad = (4 - (this.byteLength % 4)) % 4;
    if (pad) {
      this.chunks.push(new Uint8Array(pad));
      this.byteLength += pad;
    }
  }

  private concatChunks(): Uint8Array {
    const out = new Uint8Array(this.byteLength);
    let offset = 0;
    for (const chunk of this.chunks) { out.set(chunk, offset); offset += chunk.byteLength; }
    return out;
  }

  private buildGLB(json: object, binary: Uint8Array): ArrayBuffer {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
    const jsonPad = (4 - (jsonBytes.byteLength % 4)) % 4;
    const binPad = (4 - (binary.byteLength % 4)) % 4;
    const jsonLen = jsonBytes.byteLength + jsonPad;
    const binLen = binary.byteLength + binPad;
    const total = 12 + 8 + jsonLen + (binLen > 0 ? 8 + binLen : 0);

    const buffer = new ArrayBuffer(total);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let o = 0;
    view.setUint32(o, GLB_MAGIC, true); o += 4;
    view.setUint32(o, 2, true); o += 4;
    view.setUint32(o, total, true); o += 4;

    view.setUint32(o, jsonLen, true); o += 4;
    view.setUint32(o, CHUNK_JSON, true); o += 4;
    bytes.set(jsonBytes, o); o += jsonBytes.byteLength;
    for (let i = 0; i < jsonPad; i++) bytes[o++] = 0x20;

    if (binLen > 0) {
      view.setUint32(o, binLen, true); o += 4;
      view.setUint32(o, CHUNK_BIN, true); o += 4;
      bytes.set(binary, o); o += binary.byteLength;
      o += binPad;
    }
    return buffer;
  }
}

/** Encode an ImageBitmap/Canvas/OffscreenCanvas to PNG bytes. Returns empty on failure. */
async function encodeSource(source: NonNullable<Texture['source']>): Promise<Uint8Array> {
  try {
    if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) {
      return await new Promise<Uint8Array>((resolve) => {
        source.toBlob((blob) => {
          if (!blob) { resolve(new Uint8Array(0)); return; }
          blob.arrayBuffer().then((ab) => resolve(new Uint8Array(ab)));
        }, 'image/png');
      });
    }
    if (typeof OffscreenCanvas !== 'undefined' && source instanceof OffscreenCanvas) {
      const blob = await source.convertToBlob({ type: 'image/png' });
      return new Uint8Array(await blob.arrayBuffer());
    }
    // ImageBitmap: draw onto a temporary OffscreenCanvas
    if (typeof OffscreenCanvas !== 'undefined') {
      const bmp = source as ImageBitmap;
      const canvas = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = canvas.getContext('2d');
      if (!ctx) return new Uint8Array(0);
      ctx.drawImage(bmp, 0, 0);
      const blob = await canvas.convertToBlob({ type: 'image/png' });
      return new Uint8Array(await blob.arrayBuffer());
    }
  } catch {
    // silently skip unserializable sources
  }
  return new Uint8Array(0);
}

function glWrap(mode: WrapMode): number {
  if (mode === 'clamp') return 33071;
  if (mode === 'mirror') return 33648;
  return 10497; // repeat
}

function glFilter(mode: FilterMode): number {
  return mode === 'nearest' ? 9728 : 9729;
}

/** Base64-encode bytes via the platform's `btoa` (or Node's `Buffer` as a fallback). */
function base64Encode(bytes: Uint8Array): string {
  if (typeof btoa === 'function') {
    let binary = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }
  const nodeBuffer = (globalThis as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } }).Buffer;
  return nodeBuffer!.from(bytes).toString('base64');
}
