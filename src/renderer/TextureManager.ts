import { Texture, type WrapMode, type FilterMode } from '../textures/Texture';
import { DataTexture } from '../textures/DataTexture';
import { MipmapGenerator } from './MipmapGenerator';

interface GPUTextureEntry {
  texture: GPUTexture;
  view: GPUTextureView;
  sampler: GPUSampler;
  version: number;
}

function wrapToGPU(w: WrapMode): GPUAddressMode {
  switch (w) {
    case 'clamp': return 'clamp-to-edge';
    case 'mirror': return 'mirror-repeat';
    default: return 'repeat';
  }
}

function filterToGPU(f: FilterMode): GPUFilterMode {
  return f === 'nearest' ? 'nearest' : 'linear';
}

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

/** Convert one float32 to a float16 (half) bit pattern. */
export function floatToHalf(value: number): number {
  _f32[0] = value;
  const x = _i32[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) return sign; // underflow → ±0 (subnormals flushed)
  if (exp >= 0x1f) return sign | 0x7c00; // overflow/inf → ±inf
  return sign | (exp << 10) | (mant >> 13);
}

/** Convert a float32 array to an equivalent half-float (Uint16) array. */
export function floatToHalfArray(src: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = floatToHalf(src[i]);
  return out;
}

/** Uploads engine Textures to the GPU and provides shared default resources. */
export class TextureManager {
  private cache = new WeakMap<Texture, GPUTextureEntry>();
  private mipmaps: MipmapGenerator;

  readonly defaultWhiteView: GPUTextureView;
  readonly defaultNormalView: GPUTextureView;
  readonly defaultSampler: GPUSampler;

  constructor(private device: GPUDevice) {
    this.mipmaps = new MipmapGenerator(device);
    this.defaultSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    this.defaultWhiteView = this.createSolid([255, 255, 255, 255], 'rgba8unorm');
    this.defaultNormalView = this.createSolid([128, 128, 255, 255], 'rgba8unorm');
  }

  /** Upload a DataTexture: float → rgba16float, uint8 → rgba8unorm(/-srgb). */
  private createData(texture: DataTexture): GPUTextureEntry {
    const { width, height, data } = texture;
    const isFloat = data instanceof Float32Array;
    const format: GPUTextureFormat = isFloat
      ? 'rgba16float'
      : texture.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm';
    const mipLevelCount = texture.generateMipmaps ? MipmapGenerator.mipLevelCount(width, height) : 1;

    const gpuTexture = this.device.createTexture({
      size: [width, height],
      format,
      mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const bytes: ArrayBufferView = isFloat ? floatToHalfArray(data as Float32Array) : (data as Uint8Array);
    const bytesPerRow = width * (isFloat ? 8 : 4); // rgba16 = 8, rgba8 = 4
    this.device.queue.writeTexture(
      { texture: gpuTexture },
      bytes as ArrayBuffer & ArrayBufferView,
      { bytesPerRow, rowsPerImage: height },
      { width, height },
    );
    if (mipLevelCount > 1) this.mipmaps.generate(gpuTexture, format);

    const sampler = this.device.createSampler({
      addressModeU: wrapToGPU(texture.wrapS),
      addressModeV: wrapToGPU(texture.wrapT),
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    });
    return { texture: gpuTexture, view: gpuTexture.createView(), sampler, version: texture.version };
  }

  private createSolid(rgba: number[], format: GPUTextureFormat): GPUTextureView {
    const texture = this.device.createTexture({
      size: [1, 1],
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture },
      new Uint8Array(rgba),
      { bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    return texture.createView();
  }

  get(texture: Texture): GPUTextureEntry {
    const existing = this.cache.get(texture);
    if (existing && existing.version === texture.version) return existing;
    if (existing) existing.texture.destroy();

    const entry = this.create(texture);
    this.cache.set(texture, entry);
    return entry;
  }

  private create(texture: Texture): GPUTextureEntry {
    if (texture instanceof DataTexture) return this.createData(texture);
    const source = texture.source;
    if (!source) throw new Error('[vela] texture has no source');

    const width = source.width;
    const height = source.height;
    const format: GPUTextureFormat =
      texture.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm';

    const mipLevelCount = texture.generateMipmaps
      ? MipmapGenerator.mipLevelCount(width, height)
      : 1;

    const gpuTexture = this.device.createTexture({
      size: [width, height],
      format,
      mipLevelCount,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source, flipY: texture.flipY },
      { texture: gpuTexture },
      { width, height },
    );

    if (mipLevelCount > 1) {
      this.mipmaps.generate(gpuTexture, format);
    }

    const mag = filterToGPU(texture.magFilter);
    const min = filterToGPU(texture.minFilter);
    // Anisotropic filtering is only valid when all filters are linear.
    const allLinear = mag === 'linear' && min === 'linear';
    const sampler = this.device.createSampler({
      addressModeU: wrapToGPU(texture.wrapS),
      addressModeV: wrapToGPU(texture.wrapT),
      magFilter: mag,
      minFilter: min,
      mipmapFilter: 'linear',
      maxAnisotropy: allLinear ? 16 : 1,
    });

    return {
      texture: gpuTexture,
      view: gpuTexture.createView(),
      sampler,
      version: texture.version,
    };
  }
}
