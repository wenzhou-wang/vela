import { Texture, type TextureOptions } from './Texture';

/**
 * A texture backed by raw pixel data (rather than an ImageBitmap). Used for HDR
 * environments: the renderer uploads it as `rgba16float`. Always linear.
 */
export class DataTexture extends Texture {
  readonly isDataTexture = true;
  /** Interleaved RGBA float data, length = width * height * 4. */
  data: Float32Array;
  width: number;
  height: number;

  constructor(data: Float32Array, width: number, height: number, options: TextureOptions = {}) {
    super(null, options);
    this.data = data;
    this.width = width;
    this.height = height;
    this.colorSpace = 'linear';
  }
}
