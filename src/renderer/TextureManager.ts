import { Texture, type WrapMode, type FilterMode } from '../textures/Texture';
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
