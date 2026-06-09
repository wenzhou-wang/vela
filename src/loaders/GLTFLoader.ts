import { Object3D } from '../core/Object3D';
import { Mesh } from '../core/Mesh';
import { SkinnedMesh } from '../core/SkinnedMesh';
import { Skeleton } from '../core/Skeleton';
import { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';
import { StandardMaterial } from '../materials/StandardMaterial';
import { Texture } from '../textures/Texture';
import { Box3 } from '../math/Box3';
import { Matrix4 } from '../math/Matrix4';
import { Vector3 } from '../math/Vector3';
import { computeTangents } from './computeTangents';
import type { MeshoptDecoder, KTX2TextureLoader } from './decoders';
import { AnimationClip } from '../animation/AnimationClip';
import { KeyframeTrack, type TrackPath, type InterpolationMode } from '../animation/KeyframeTrack';
import {
  type GLTFRoot,
  type GLTFMaterial,
  type GLTFNode,
  type GLTFPrimitive,
  COMPONENT_SIZE,
  TYPE_COMPONENTS,
} from './gltfTypes';

export interface GLTFResult {
  scene: Object3D;
  /** All top-level scenes' roots; `scene` is the default. */
  scenes: Object3D[];
  materials: StandardMaterial[];
  /** Keyframe animation clips targeting the loaded nodes. */
  animations: AnimationClip[];
  /** World-space bounding box of the loaded content. */
  boundingBox: Box3;
}

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"

/**
 * Loads glTF 2.0 (.gltf + .bin + images) and binary glTF (.glb).
 * Supports node hierarchy, meshes/primitives, PBR metallic-roughness materials,
 * textures/samplers, skinning, morph targets, keyframe animation
 * (translation/rotation/scale/weights), and KHR_materials_emissive_strength.
 */
export class GLTFLoader {
  private meshoptDecoder: MeshoptDecoder | null = null;
  private ktx2Loader: KTX2TextureLoader | null = null;

  /** Provide a meshoptimizer decoder to enable `EXT_meshopt_compression`. */
  setMeshoptDecoder(decoder: MeshoptDecoder): this {
    this.meshoptDecoder = decoder;
    return this;
  }

  /** Provide a KTX2 loader/transcoder to enable `KHR_texture_basisu`. */
  setKTX2Loader(loader: KTX2TextureLoader): this {
    this.ktx2Loader = loader;
    return this;
  }

  /** Load from a URL (resolves relative buffers/images against it). */
  async load(url: string): Promise<GLTFResult> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`[vela] failed to fetch ${url}: ${response.status}`);
    const buffer = await response.arrayBuffer();
    const baseUrl = url.substring(0, url.lastIndexOf('/') + 1);
    return this.parse(buffer, baseUrl);
  }

  /** Parse an ArrayBuffer that is either a .glb or a UTF-8 .gltf JSON. */
  async parse(data: ArrayBuffer, baseUrl = ''): Promise<GLTFResult> {
    const view = new DataView(data);
    let json: GLTFRoot;
    let glbBinary: Uint8Array | null = null;

    if (view.byteLength >= 12 && view.getUint32(0, true) === GLB_MAGIC) {
      ({ json, glbBinary } = this.parseGLB(data));
    } else {
      json = JSON.parse(new TextDecoder().decode(data)) as GLTFRoot;
    }

    return this.build(json, baseUrl, glbBinary);
  }

  private parseGLB(data: ArrayBuffer): { json: GLTFRoot; glbBinary: Uint8Array | null } {
    const view = new DataView(data);
    const length = view.getUint32(8, true);
    let offset = 12;
    let json: GLTFRoot | null = null;
    let glbBinary: Uint8Array | null = null;

    while (offset < length) {
      const chunkLength = view.getUint32(offset, true);
      const chunkType = view.getUint32(offset + 4, true);
      const chunkData = new Uint8Array(data, offset + 8, chunkLength);
      if (chunkType === CHUNK_JSON) {
        json = JSON.parse(new TextDecoder().decode(chunkData)) as GLTFRoot;
      } else if (chunkType === CHUNK_BIN) {
        glbBinary = chunkData;
      }
      offset += 8 + chunkLength;
    }
    if (!json) throw new Error('[vela] GLB has no JSON chunk');
    return { json, glbBinary };
  }

  private async build(
    json: GLTFRoot,
    baseUrl: string,
    glbBinary: Uint8Array | null,
  ): Promise<GLTFResult> {
    const buffers = await this.loadBuffers(json, baseUrl, glbBinary);
    const meshopt = await this.decodeMeshopt(json, buffers);
    const images = await this.loadImages(json, baseUrl, buffers);
    const textures = await this.buildTextures(json, images, buffers);
    const materials = (json.materials ?? []).map((m) => this.buildMaterial(m, textures));
    if (materials.length === 0) materials.push(new StandardMaterial({ color: 0xcccccc, metalness: 0.1, roughness: 0.8 }));

    const ctx: BuildContext = { json, buffers, materials, geometryCache: new Map(), pendingSkins: [], meshopt };

    // Build nodes
    const nodes: Object3D[] = (json.nodes ?? []).map((n) => this.buildNode(n, ctx));

    // Wire up hierarchy
    (json.nodes ?? []).forEach((n, i) => {
      if (n.children) {
        for (const childIndex of n.children) nodes[i].add(nodes[childIndex]);
      }
    });

    const sceneRoots: Object3D[] = [];
    const scenes = json.scenes ?? [{ nodes: nodes.map((_, i) => i) }];
    for (const s of scenes) {
      const root = new Object3D();
      root.name = s.name ?? 'Scene';
      for (const nodeIndex of s.nodes ?? []) root.add(nodes[nodeIndex]);
      sceneRoots.push(root);
    }

    // Resolve skinned meshes now that all joint nodes exist.
    const skeletonCache = new Map<number, Skeleton>();
    for (const { mesh, skinIndex } of ctx.pendingSkins) {
      let skeleton = skeletonCache.get(skinIndex);
      if (!skeleton) {
        skeleton = this.buildSkeleton(skinIndex, nodes, ctx);
        skeletonCache.set(skinIndex, skeleton);
      }
      mesh.skeleton = skeleton;
    }

    const defaultScene = sceneRoots[json.scene ?? 0] ?? sceneRoots[0] ?? new Object3D();
    defaultScene.updateMatrixWorld(true);

    const animations = this.buildAnimations(json, nodes, ctx);
    const boundingBox = this.computeBounds(defaultScene);

    return { scene: defaultScene, scenes: sceneRoots, materials, animations, boundingBox };
  }

  private buildAnimations(json: GLTFRoot, nodes: Object3D[], ctx: BuildContext): AnimationClip[] {
    const clips: AnimationClip[] = [];
    json.animations?.forEach((anim, ai) => {
      const tracks: KeyframeTrack[] = [];
      for (const channel of anim.channels) {
        const path = channel.target.path;
        const nodeIndex = channel.target.node;
        if (nodeIndex === undefined) continue;
        const target = nodes[nodeIndex];
        const sampler = anim.samplers[channel.sampler];
        const times = this.readAccessorFloat(ctx, sampler.input);
        const values = this.readAccessorFloat(ctx, sampler.output);
        const interpolation = (sampler.interpolation ?? 'LINEAR') as InterpolationMode;

        if (path === 'weights') {
          // glTF animates the node's mesh; here that mesh lives as child meshes.
          // stride = morph-target count, derived from each mesh's influences.
          for (const child of target.children) {
            if (child instanceof Mesh && child.morphTargetInfluences.length > 0) {
              tracks.push(
                new KeyframeTrack(child, 'weights', times, values, interpolation, child.morphTargetInfluences.length),
              );
            }
          }
          continue;
        }

        tracks.push(new KeyframeTrack(target, path as TrackPath, times, values, interpolation));
      }
      if (tracks.length) clips.push(new AnimationClip(anim.name ?? `clip_${ai}`, tracks));
    });
    return clips;
  }

  private buildNode(node: GLTFNode, ctx: BuildContext): Object3D {
    const hasMesh = node.mesh !== undefined;
    const object = new Object3D();
    object.name = node.name ?? '';

    if (node.matrix) {
      object.matrix.fromArray(node.matrix);
      object.matrix.decompose(object.position, object.quaternion, object.scale);
    } else {
      if (node.translation) object.position.fromArray(node.translation);
      if (node.rotation) object.quaternion.fromArray(node.rotation);
      if (node.scale) object.scale.fromArray(node.scale);
    }

    if (hasMesh) {
      const meshDef = ctx.json.meshes![node.mesh!];
      for (let p = 0; p < meshDef.primitives.length; p++) {
        const prim = meshDef.primitives[p];
        if (prim.mode !== undefined && prim.mode !== 4) continue; // only TRIANGLES
        const geometry = this.buildGeometry(prim, ctx);
        const material = prim.material !== undefined ? ctx.materials[prim.material] : ctx.materials[ctx.materials.length - 1];
        if (material.normalMap && !geometry.getAttribute('tangent')) {
          computeTangents(geometry);
        }
        let mesh: Mesh;
        if (node.skin !== undefined) {
          // placeholder skeleton; resolved after all nodes are built
          const skinned = new SkinnedMesh(geometry, material, new Skeleton([], []));
          ctx.pendingSkins.push({ mesh: skinned, skinIndex: node.skin });
          mesh = skinned;
        } else {
          mesh = new Mesh(geometry, material);
        }
        mesh.name = meshDef.name ? `${meshDef.name}_${p}` : object.name;

        // Morph-target weights & names (mesh.weights are the defaults).
        if (geometry.morphAttributes.position?.length) {
          const n = geometry.morphAttributes.position.length;
          if (meshDef.weights && meshDef.weights.length === n) {
            mesh.morphTargetInfluences = meshDef.weights.slice();
          }
          const names = meshDef.extras?.targetNames;
          if (names && names.length === n) {
            mesh.morphTargetDictionary = {};
            names.forEach((name, i) => (mesh.morphTargetDictionary![name] = i));
          }
        }

        object.add(mesh);
      }
    }

    return object;
  }

  private buildSkeleton(skinIndex: number, nodes: Object3D[], ctx: BuildContext): Skeleton {
    const skin = ctx.json.skins![skinIndex];
    const joints = skin.joints.map((i) => nodes[i]);
    let inverses: Matrix4[];
    if (skin.inverseBindMatrices !== undefined) {
      const data = this.readAccessorFloat(ctx, skin.inverseBindMatrices); // 16 floats per joint
      inverses = joints.map((_, j) => new Matrix4().fromArray(data, j * 16));
    } else {
      inverses = joints.map(() => new Matrix4());
    }
    return new Skeleton(joints, inverses);
  }

  private buildGeometry(prim: GLTFPrimitive, ctx: BuildContext): BufferGeometry {
    const key = JSON.stringify(prim);
    const cached = ctx.geometryCache.get(key);
    if (cached) return cached;

    const geometry = new BufferGeometry();
    const attributes = prim.attributes as Record<string, number>;

    if (attributes.POSITION !== undefined) {
      geometry.setAttribute('position', new BufferAttribute(this.readAccessorFloat(ctx, attributes.POSITION), 3));
    }
    if (attributes.NORMAL !== undefined) {
      geometry.setAttribute('normal', new BufferAttribute(this.readAccessorFloat(ctx, attributes.NORMAL), 3));
    }
    if (attributes.TEXCOORD_0 !== undefined) {
      geometry.setAttribute('uv', new BufferAttribute(this.readAccessorFloat(ctx, attributes.TEXCOORD_0), 2));
    }
    if (attributes.TANGENT !== undefined) {
      geometry.setAttribute('tangent', new BufferAttribute(this.readAccessorFloat(ctx, attributes.TANGENT), 4));
    }
    if (attributes.COLOR_0 !== undefined) {
      // COLOR_0 may be VEC3 or VEC4; the renderer expects vec4, so pad RGB with a=1.
      let color = this.readAccessorFloat(ctx, attributes.COLOR_0);
      if (ctx.json.accessors![attributes.COLOR_0].type === 'VEC3') {
        const rgba = new Float32Array((color.length / 3) * 4);
        for (let i = 0, j = 0; i < color.length; i += 3, j += 4) {
          rgba[j] = color[i]; rgba[j + 1] = color[i + 1]; rgba[j + 2] = color[i + 2]; rgba[j + 3] = 1;
        }
        color = rgba;
      }
      geometry.setAttribute('color', new BufferAttribute(color, 4));
    }
    if (attributes.JOINTS_0 !== undefined) {
      geometry.setAttribute('joints', new BufferAttribute(this.readAccessorUint(ctx, attributes.JOINTS_0), 4));
    }
    if (attributes.WEIGHTS_0 !== undefined) {
      geometry.setAttribute('weights', new BufferAttribute(this.readAccessorFloat(ctx, attributes.WEIGHTS_0), 4));
    }

    if (prim.targets && prim.targets.length) {
      const positions: BufferAttribute[] = [];
      const normals: BufferAttribute[] = [];
      let allHaveNormals = true;
      for (const target of prim.targets) {
        if (target.POSITION !== undefined) {
          positions.push(new BufferAttribute(this.readAccessorFloat(ctx, target.POSITION), 3));
        }
        if (target.NORMAL !== undefined) {
          normals.push(new BufferAttribute(this.readAccessorFloat(ctx, target.NORMAL), 3));
        } else {
          allHaveNormals = false;
        }
      }
      if (positions.length === prim.targets.length) geometry.morphAttributes.position = positions;
      if (allHaveNormals && normals.length === prim.targets.length) geometry.morphAttributes.normal = normals;
    }

    if (prim.indices !== undefined) {
      geometry.setIndex(this.readIndices(ctx, prim.indices));
    }

    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    ctx.geometryCache.set(key, geometry);
    return geometry;
  }

  /**
   * Decode every `EXT_meshopt_compression` buffer view into a tightly-packed
   * Uint8Array via the supplied meshopt decoder. Throws if the extension is used
   * without a decoder.
   */
  private async decodeMeshopt(
    json: GLTFRoot,
    buffers: Uint8Array[],
  ): Promise<Map<number, { data: Uint8Array; byteStride: number }>> {
    const out = new Map<number, { data: Uint8Array; byteStride: number }>();
    const views = json.bufferViews ?? [];
    if (!views.some((v) => v.extensions?.EXT_meshopt_compression)) return out;

    const decoder = this.meshoptDecoder;
    if (!decoder) {
      throw new Error('[vela] EXT_meshopt_compression requires a decoder — call setMeshoptDecoder(MeshoptDecoder)');
    }
    await decoder.ready;

    for (let i = 0; i < views.length; i++) {
      const ext = views[i].extensions?.EXT_meshopt_compression;
      if (!ext) continue;
      const src = buffers[ext.buffer].subarray(ext.byteOffset ?? 0, (ext.byteOffset ?? 0) + ext.byteLength);
      const target = new Uint8Array(ext.count * ext.byteStride);
      decoder.decodeGltfBuffer(target, ext.count, ext.byteStride, src, ext.mode, ext.filter ?? 'NONE');
      out.set(i, { data: target, byteStride: ext.byteStride });
    }
    return out;
  }

  /** Resolve a buffer view to its bytes, honoring meshopt-decoded views. */
  private resolveView(ctx: BuildContext, bufferViewIndex: number): { array: Uint8Array; byteOffset: number; byteStride?: number } {
    const decoded = ctx.meshopt.get(bufferViewIndex);
    if (decoded) return { array: decoded.data, byteOffset: 0, byteStride: decoded.byteStride };
    const bv = ctx.json.bufferViews![bufferViewIndex];
    return { array: ctx.buffers[bv.buffer], byteOffset: bv.byteOffset ?? 0, byteStride: bv.byteStride };
  }

  private readIndices(ctx: BuildContext, accessorIndex: number): Uint16Array | Uint32Array {
    const accessor = ctx.json.accessors![accessorIndex];
    const view = this.resolveView(ctx, accessor.bufferView!);
    const buffer = view.array;
    const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const compSize = COMPONENT_SIZE[accessor.componentType];
    const base = view.byteOffset + (accessor.byteOffset ?? 0);
    const count = accessor.count;

    if (accessor.componentType === 5125) {
      const out = new Uint32Array(count);
      for (let i = 0; i < count; i++) out[i] = dv.getUint32(base + i * compSize, true);
      return out;
    }
    const out = new Uint16Array(count);
    if (accessor.componentType === 5121) {
      for (let i = 0; i < count; i++) out[i] = dv.getUint8(base + i);
    } else {
      for (let i = 0; i < count; i++) out[i] = dv.getUint16(base + i * compSize, true);
    }
    return out;
  }

  /** Read any accessor into a tightly-packed Float32Array (normalizing ints). */
  private readAccessorFloat(ctx: BuildContext, accessorIndex: number): Float32Array {
    const accessor = ctx.json.accessors![accessorIndex];
    const numComponents = TYPE_COMPONENTS[accessor.type];
    const out = new Float32Array(accessor.count * numComponents);

    if (accessor.bufferView === undefined) return out; // sparse-only / empty
    const view = this.resolveView(ctx, accessor.bufferView);
    const buffer = view.array;
    const compSize = COMPONENT_SIZE[accessor.componentType];
    const byteStride = view.byteStride ?? numComponents * compSize;
    const baseOffset = view.byteOffset + (accessor.byteOffset ?? 0);
    const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    const normalize = accessor.normalized ?? false;
    for (let i = 0; i < accessor.count; i++) {
      const elementOffset = baseOffset + i * byteStride;
      for (let c = 0; c < numComponents; c++) {
        const o = elementOffset + c * compSize;
        out[i * numComponents + c] = this.readComponent(dv, o, accessor.componentType, normalize);
      }
    }
    return out;
  }

  /** Read an integer accessor (e.g. JOINTS_0) into a Uint32Array. */
  private readAccessorUint(ctx: BuildContext, accessorIndex: number): Uint32Array {
    const accessor = ctx.json.accessors![accessorIndex];
    const numComponents = TYPE_COMPONENTS[accessor.type];
    const out = new Uint32Array(accessor.count * numComponents);
    if (accessor.bufferView === undefined) return out;
    const view = this.resolveView(ctx, accessor.bufferView);
    const buffer = view.array;
    const compSize = COMPONENT_SIZE[accessor.componentType];
    const byteStride = view.byteStride ?? numComponents * compSize;
    const baseOffset = view.byteOffset + (accessor.byteOffset ?? 0);
    const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    for (let i = 0; i < accessor.count; i++) {
      const elementOffset = baseOffset + i * byteStride;
      for (let c = 0; c < numComponents; c++) {
        const o = elementOffset + c * compSize;
        out[i * numComponents + c] =
          accessor.componentType === 5123 ? dv.getUint16(o, true) : dv.getUint8(o);
      }
    }
    return out;
  }

  private readComponent(dv: DataView, offset: number, componentType: number, normalize: boolean): number {
    switch (componentType) {
      case 5126: return dv.getFloat32(offset, true);
      case 5125: return dv.getUint32(offset, true);
      case 5123: { const v = dv.getUint16(offset, true); return normalize ? v / 65535 : v; }
      case 5122: { const v = dv.getInt16(offset, true); return normalize ? Math.max(v / 32767, -1) : v; }
      case 5121: { const v = dv.getUint8(offset); return normalize ? v / 255 : v; }
      case 5120: { const v = dv.getInt8(offset); return normalize ? Math.max(v / 127, -1) : v; }
      default: return 0;
    }
  }

  private async loadBuffers(json: GLTFRoot, baseUrl: string, glbBinary: Uint8Array | null): Promise<Uint8Array[]> {
    const buffers: Uint8Array[] = [];
    for (const buffer of json.buffers ?? []) {
      if (buffer.uri === undefined) {
        if (!glbBinary) throw new Error('[vela] buffer without uri but no GLB binary chunk');
        buffers.push(glbBinary);
      } else if (buffer.uri.startsWith('data:')) {
        buffers.push(decodeDataURI(buffer.uri));
      } else {
        const res = await fetch(baseUrl + decodeURIComponent(buffer.uri));
        buffers.push(new Uint8Array(await res.arrayBuffer()));
      }
    }
    return buffers;
  }

  private async loadImages(json: GLTFRoot, baseUrl: string, buffers: Uint8Array[]): Promise<(ImageBitmap | null)[]> {
    const out: (ImageBitmap | null)[] = [];
    for (const image of json.images ?? []) {
      let blob: Blob;
      if (image.uri && image.uri.startsWith('data:')) {
        blob = new Blob([decodeDataURI(image.uri) as BlobPart]);
      } else if (image.uri) {
        const res = await fetch(baseUrl + decodeURIComponent(image.uri));
        blob = await res.blob();
      } else if (image.bufferView !== undefined) {
        const bv = json.bufferViews![image.bufferView];
        const buf = buffers[bv.buffer];
        const slice = buf.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
        blob = new Blob([slice as BlobPart], { type: image.mimeType ?? 'image/png' });
      } else {
        out.push(null);
        continue;
      }
      try {
        out.push(await createImageBitmap(blob, { colorSpaceConversion: 'none' }));
      } catch (e) {
        console.warn('[vela] failed to decode image', e);
        out.push(null);
      }
    }
    return out;
  }

  private async buildTextures(
    json: GLTFRoot,
    images: (ImageBitmap | null)[],
    buffers: Uint8Array[],
  ): Promise<Texture[]> {
    const samplers = json.samplers ?? [];
    const out: Texture[] = [];
    for (const tex of json.textures ?? []) {
      // KHR_texture_basisu: the source is a KTX2 image to transcode.
      const basisuSource = tex.extensions?.KHR_texture_basisu?.source;
      let texture: Texture;
      if (basisuSource !== undefined) {
        if (!this.ktx2Loader) {
          throw new Error('[vela] KHR_texture_basisu requires a KTX2 loader — call setKTX2Loader(new KTX2Loader())');
        }
        const bytes = this.imageBytes(json, basisuSource, buffers);
        texture = await this.ktx2Loader.parse(bytes, false);
      } else {
        const image = tex.source !== undefined ? images[tex.source] : null;
        texture = new Texture(image);
      }
      const sampler = tex.sampler !== undefined ? samplers[tex.sampler] : undefined;
      if (sampler) {
        texture.wrapS = wrapMode(sampler.wrapS);
        texture.wrapT = wrapMode(sampler.wrapT);
        if (sampler.magFilter === 9728) texture.magFilter = 'nearest';
        if (sampler.minFilter === 9728 || sampler.minFilter === 9984 || sampler.minFilter === 9986) texture.minFilter = 'nearest';
      }
      out.push(texture);
    }
    return out;
  }

  /** Raw bytes for an image (data URI, external URI, or buffer view). */
  private imageBytes(json: GLTFRoot, imageIndex: number, buffers: Uint8Array[]): Uint8Array {
    const image = json.images![imageIndex];
    if (image.uri && image.uri.startsWith('data:')) return decodeDataURI(image.uri);
    if (image.bufferView !== undefined) {
      const bv = json.bufferViews![image.bufferView];
      const buf = buffers[bv.buffer];
      return buf.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
    }
    throw new Error('[vela] KTX2 image must use a data URI or bufferView (external URI not supported here)');
  }

  private buildMaterial(def: GLTFMaterial, textures: Texture[]): StandardMaterial {
    const pbr = def.pbrMetallicRoughness ?? {};
    const material = new StandardMaterial();
    material.name = def.name ?? '';

    if (pbr.baseColorFactor) {
      material.color.setRGB(
        srgbToLinear(pbr.baseColorFactor[0]),
        srgbToLinear(pbr.baseColorFactor[1]),
        srgbToLinear(pbr.baseColorFactor[2]),
      );
      material.opacity = pbr.baseColorFactor[3];
    }
    material.metalness = pbr.metallicFactor ?? 1;
    material.roughness = pbr.roughnessFactor ?? 1;

    if (pbr.baseColorTexture) {
      material.map = configure(textures[pbr.baseColorTexture.index], 'srgb');
    }
    if (pbr.metallicRoughnessTexture) {
      material.metalnessRoughnessMap = configure(textures[pbr.metallicRoughnessTexture.index], 'linear');
    }
    if (def.normalTexture) {
      material.normalMap = configure(textures[def.normalTexture.index], 'linear');
      material.normalScale = def.normalTexture.scale ?? 1;
    }
    if (def.occlusionTexture) {
      material.occlusionMap = configure(textures[def.occlusionTexture.index], 'linear');
      material.occlusionStrength = def.occlusionTexture.strength ?? 1;
    }
    if (def.emissiveFactor) {
      material.emissive.setRGB(
        srgbToLinear(def.emissiveFactor[0]),
        srgbToLinear(def.emissiveFactor[1]),
        srgbToLinear(def.emissiveFactor[2]),
      );
    }
    const emissiveStrength = def.extensions?.KHR_materials_emissive_strength?.emissiveStrength;
    if (emissiveStrength !== undefined) material.emissiveIntensity = emissiveStrength;
    if (def.emissiveTexture) {
      material.emissiveMap = configure(textures[def.emissiveTexture.index], 'srgb');
    }

    const clearcoat = def.extensions?.KHR_materials_clearcoat;
    if (clearcoat) {
      material.clearcoat = clearcoat.clearcoatFactor ?? 0;
      material.clearcoatRoughness = clearcoat.clearcoatRoughnessFactor ?? 0;
      if (clearcoat.clearcoatTexture) {
        material.clearcoatMap = configure(textures[clearcoat.clearcoatTexture.index], 'linear');
      }
      if (clearcoat.clearcoatRoughnessTexture) {
        material.clearcoatRoughnessMap = configure(textures[clearcoat.clearcoatRoughnessTexture.index], 'linear');
      }
    }

    const iorExt = def.extensions?.KHR_materials_ior;
    if (iorExt?.ior !== undefined) material.ior = iorExt.ior;

    const specularExt = def.extensions?.KHR_materials_specular;
    if (specularExt) {
      if (specularExt.specularFactor !== undefined) material.specularIntensity = specularExt.specularFactor;
      const scf = specularExt.specularColorFactor;
      if (scf) material.specularColor.setRGB(scf[0], scf[1], scf[2]); // already linear
    }

    const sheenExt = def.extensions?.KHR_materials_sheen;
    if (sheenExt) {
      const scf = sheenExt.sheenColorFactor;
      if (scf) material.sheenColor.setRGB(scf[0], scf[1], scf[2]); // already linear
      if (sheenExt.sheenRoughnessFactor !== undefined) material.sheenRoughness = sheenExt.sheenRoughnessFactor;
    }

    const transmissionExt = def.extensions?.KHR_materials_transmission;
    if (transmissionExt?.transmissionFactor !== undefined) material.transmission = transmissionExt.transmissionFactor;

    const volumeExt = def.extensions?.KHR_materials_volume;
    if (volumeExt) {
      if (volumeExt.thicknessFactor !== undefined) material.thickness = volumeExt.thicknessFactor;
      if (volumeExt.attenuationDistance !== undefined) material.attenuationDistance = volumeExt.attenuationDistance;
      const ac = volumeExt.attenuationColor;
      if (ac) material.attenuationColor.setRGB(ac[0], ac[1], ac[2]);
    }

    if (def.doubleSided) material.side = 'double';
    if (def.alphaMode === 'BLEND') {
      material.transparent = true;
      material.depthWrite = false;
    } else if (def.alphaMode === 'MASK') {
      material.alphaTest = def.alphaCutoff ?? 0.5;
    }

    return material;
  }

  private computeBounds(root: Object3D): Box3 {
    const box = new Box3();
    const tmp = new Box3();
    const v = new Vector3();
    root.traverse((object) => {
      if (object instanceof Mesh) {
        const geo = object.geometry;
        if (!geo.boundingBox) geo.computeBoundingBox();
        if (geo.boundingBox && !geo.boundingBox.isEmpty()) {
          tmp.copy(geo.boundingBox).applyMatrix4(object.matrixWorld);
          box.union(tmp);
        }
      }
    });
    if (box.isEmpty()) box.expandByPoint(v.set(0, 0, 0));
    return box;
  }
}

interface BuildContext {
  json: GLTFRoot;
  buffers: Uint8Array[];
  materials: StandardMaterial[];
  geometryCache: Map<string, BufferGeometry>;
  /** Skinned meshes awaiting skeleton resolution (joints built after nodes). */
  pendingSkins: { mesh: SkinnedMesh; skinIndex: number }[];
  /** Decoded EXT_meshopt_compression buffer views, by bufferView index. */
  meshopt: Map<number, { data: Uint8Array; byteStride: number }>;
}

function configure(texture: Texture | undefined, colorSpace: 'srgb' | 'linear'): Texture | null {
  if (!texture) return null;
  texture.colorSpace = colorSpace;
  return texture;
}

function wrapMode(v?: number): 'repeat' | 'clamp' | 'mirror' {
  if (v === 33071) return 'clamp';
  if (v === 33648) return 'mirror';
  return 'repeat';
}

function srgbToLinear(c: number): number {
  // glTF factors are already linear; identity kept for clarity/extension hooks
  return c;
}

function decodeDataURI(uri: string): Uint8Array {
  const comma = uri.indexOf(',');
  const meta = uri.substring(5, comma);
  const dataPart = uri.substring(comma + 1);
  if (meta.includes('base64')) {
    const binary = atob(dataPart);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }
  return new TextEncoder().encode(decodeURIComponent(dataPart));
}

