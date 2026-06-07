import { generateUUID } from '../math/MathUtils';

export type WrapMode = 'repeat' | 'clamp' | 'mirror';
export type FilterMode = 'nearest' | 'linear';

export type TextureSource = ImageBitmap | HTMLCanvasElement | OffscreenCanvas;

export interface TextureOptions {
  wrapS?: WrapMode;
  wrapT?: WrapMode;
  magFilter?: FilterMode;
  minFilter?: FilterMode;
  generateMipmaps?: boolean;
  /** sRGB-encoded color data (base color, emissive). Data textures are linear. */
  colorSpace?: 'srgb' | 'linear';
  flipY?: boolean;
}

/** A 2D texture backed by an ImageBitmap/canvas, uploaded lazily by the renderer. */
export class Texture {
  readonly id: string = generateUUID();
  readonly isTexture = true;
  name = '';

  source: TextureSource | null;
  wrapS: WrapMode = 'repeat';
  wrapT: WrapMode = 'repeat';
  magFilter: FilterMode = 'linear';
  minFilter: FilterMode = 'linear';
  generateMipmaps = true;
  colorSpace: 'srgb' | 'linear' = 'linear';
  flipY = false;

  /** Bumped to force re-upload. */
  version = 0;

  constructor(source: TextureSource | null = null, options: TextureOptions = {}) {
    this.source = source;
    Object.assign(this, options);
  }

  needsUpdate(): void {
    this.version++;
  }
}
