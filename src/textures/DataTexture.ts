import { Texture, type TextureOptions } from './Texture';

/**
 * A texture backed by raw pixel data (rather than an ImageBitmap). The renderer
 * uploads `Float32Array` data as `rgba16float` (HDR environments) and
 * `Uint8Array` data as `rgba8unorm`/`-srgb` (e.g. transcoded KTX2).
 */
export class DataTexture extends Texture {
  readonly isDataTexture = true;
  /** Interleaved RGBA data, length = width * height * 4. */
  data: Float32Array | Uint8Array;
  width: number;
  height: number;

  constructor(data: Float32Array | Uint8Array, width: number, height: number, options: TextureOptions = {}) {
    super(null, options);
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

/**
 * A GPU block-compressed texture (BC7 / ASTC 4×4 / ETC2). Uploaded directly to
 * the GPU without a pixel-format conversion; no mipmap generation is performed.
 */
export class CompressedDataTexture extends DataTexture {
  readonly isCompressedDataTexture = true;
  /**
   * The WebGPU texture format (e.g. `'bc7-rgba-unorm'`, `'astc-4x4-unorm'`,
   * `'etc2-rgba8unorm'`).
   */
  gpuFormat: string;

  constructor(
    blocks: Uint8Array,
    width: number,
    height: number,
    gpuFormat: string,
    options: TextureOptions = {},
  ) {
    super(blocks, width, height, { ...options, generateMipmaps: false });
    this.gpuFormat = gpuFormat;
  }
}
