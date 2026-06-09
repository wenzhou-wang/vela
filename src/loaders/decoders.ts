import type { Texture } from '../textures/Texture';

/**
 * Decoded geometry from a Draco-compressed mesh primitive. The `attributes` map
 * contains per-attribute typed arrays (Float32Array for float attributes,
 * Uint32Array for integer attributes such as JOINTS_0).
 */
export interface DracoGeometry {
  indices: Uint32Array;
  /** Attribute semantic → decoded array + component count. */
  attributes: Record<string, { array: Float32Array | Uint32Array; itemSize: number }>;
}

/**
 * Minimal interface for a Draco mesh decoder (for `KHR_draco_mesh_compression`).
 * Implement against the `draco3d` or `draco3dgltf` WASM module and pass an instance
 * via {@link GLTFLoader.setDracoDecoder}.
 *
 * `attributeIds` maps attribute semantics (e.g. "POSITION", "NORMAL") to the
 * Draco-internal unique IDs listed in the primitive extension's `attributes` map.
 */
export interface DracoDecoder {
  /** Resolves once the (WASM) decoder is initialized, if applicable. */
  ready?: Promise<void>;
  /**
   * Decode a Draco-compressed buffer into indices + per-attribute arrays.
   * The decoder receives the raw compressed bytes and the Draco attribute IDs
   * to extract; it returns Float32 arrays for float attributes and Uint32 arrays
   * for integer attributes (e.g. JOINTS_0).
   */
  decode(data: Uint8Array, attributeIds: Record<string, number>): DracoGeometry | Promise<DracoGeometry>;
}

/**
 * The subset of meshoptimizer's `MeshoptDecoder` that vela uses for
 * `EXT_meshopt_compression`. Pass the standard module:
 *
 * ```ts
 * import { MeshoptDecoder } from 'meshoptimizer';
 * loader.setMeshoptDecoder(MeshoptDecoder);
 * ```
 */
export interface MeshoptDecoder {
  /** Resolves once the (WASM) decoder is initialized. */
  ready: Promise<void>;
  /** Decode `count`×`size` bytes of `source` (mode/filter per the glTF EXT) into `target`. */
  decodeGltfBuffer(
    target: Uint8Array,
    count: number,
    size: number,
    source: Uint8Array,
    mode: string,
    filter?: string,
  ): void;
}

/**
 * Transcodes a KTX2 payload into a renderable {@link Texture} (used for
 * `KHR_texture_basisu`). {@link KTX2Loader} implements this; Basis-supercompressed
 * payloads need a transcoder plugged into it.
 */
export interface KTX2TextureLoader {
  parse(data: Uint8Array, srgb: boolean): Texture | Promise<Texture>;
}
