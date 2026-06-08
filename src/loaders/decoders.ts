import type { Texture } from '../textures/Texture';

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
