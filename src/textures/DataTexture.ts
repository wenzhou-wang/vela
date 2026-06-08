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
