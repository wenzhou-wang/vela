import { DataTexture } from '../textures/DataTexture';
import { CompressedDataTexture } from '../textures/DataTexture';
import type { Texture, TextureOptions } from '../textures/Texture';
import type { KTX2TextureLoader } from './decoders';

/** A decoded KTX2 level: tightly-packed RGBA8 pixels. */
export interface KTX2Image {
  width: number;
  height: number;
  rgba: Uint8Array;
}

/** A KTX2 level transcoded to GPU-compressed blocks. */
export interface CompressedKTX2Image {
  width: number;
  height: number;
  /** Raw block-compressed data (BC7 / ASTC 4×4 / ETC2 blocks). */
  blocks: Uint8Array;
  /**
   * The WebGPU texture format (base, without sRGB suffix — the loader applies
   * the correct suffix based on the `srgb` flag).
   */
  format: 'bc7-rgba-unorm' | 'astc-4x4-unorm' | 'etc2-rgba8unorm';
}

/**
 * Transcodes a Basis-supercompressed KTX2 payload to RGBA8. Wrap the standard
 * Basis/KTX2 transcoder (which can transcode to `RGBA32`) and plug it in via
 * {@link KTX2Loader.setTranscoder}.
 */
export interface BasisTranscoder {
  ready?: Promise<void>;
  transcodeToRGBA(ktx2: Uint8Array): KTX2Image | Promise<KTX2Image>;
  /**
   * Optional: transcode to a GPU-compressed format. Only called when the device
   * supports the requested format. The `format` is the unorm base name; the
   * loader applies the sRGB suffix separately.
   */
  transcodeToCompressed?(
    ktx2: Uint8Array,
    format: 'bc7-rgba-unorm' | 'astc-4x4-unorm' | 'etc2-rgba8unorm',
  ): CompressedKTX2Image | Promise<CompressedKTX2Image>;
}

// KTX2 identifier: «´»KTX 20«bb»\r\n\x1A\n
const KTX2_ID = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];
// VkFormat values we can upload without transcoding.
const VK_R8G8B8A8_UNORM = 37;
const VK_R8G8B8A8_SRGB = 43;
const VK_UNDEFINED = 0;

interface KTX2Container {
  vkFormat: number;
  width: number;
  height: number;
  supercompression: number;
  /** Level 0 byte range (largest mip). */
  levelData: Uint8Array;
}

// Priority-ordered list of compressed formats and their feature flags.
const COMPRESSED_FORMATS: { format: 'bc7-rgba-unorm' | 'astc-4x4-unorm' | 'etc2-rgba8unorm'; feature: string }[] = [
  { format: 'bc7-rgba-unorm', feature: 'texture-compression-bc' },
  { format: 'astc-4x4-unorm', feature: 'texture-compression-astc' },
  { format: 'etc2-rgba8unorm', feature: 'texture-compression-etc2' },
];

/**
 * Loads KTX2 textures (for `KHR_texture_basisu`). Uncompressed RGBA8 payloads are
 * uploaded directly; Basis-supercompressed payloads need a transcoder
 * ({@link setTranscoder}) that yields RGBA8. When a GPUDevice is registered via
 * {@link setDevice} and the transcoder implements `transcodeToCompressed`, the loader
 * will transcode to BC7/ASTC/ETC2 (whichever the device prefers) instead of RGBA8.
 */
export class KTX2Loader implements KTX2TextureLoader {
  private transcoder: BasisTranscoder | null = null;
  private device: GPUDevice | null = null;

  setTranscoder(transcoder: BasisTranscoder): this {
    this.transcoder = transcoder;
    return this;
  }

  /**
   * Provide the WebGPU device so the loader can query feature support and
   * upload compressed textures using the preferred native format.
   */
  setDevice(device: GPUDevice): this {
    this.device = device;
    return this;
  }

  async load(url: string, options?: TextureOptions): Promise<Texture> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`[vela] failed to load KTX2: ${url} (${res.status})`);
    return this.parse(new Uint8Array(await res.arrayBuffer()), false, options);
  }

  /** Parse + (if needed) transcode KTX2 bytes into a `DataTexture`. */
  async parse(data: Uint8Array, srgb = false, options: TextureOptions = {}): Promise<Texture> {
    const c = parseContainer(data);

    const isSrgb = srgb || c.vkFormat === VK_R8G8B8A8_SRGB;

    // Uncompressed RGBA8: upload directly (no transcoding needed).
    if (c.supercompression === 0 && (c.vkFormat === VK_R8G8B8A8_UNORM || c.vkFormat === VK_R8G8B8A8_SRGB)) {
      return new DataTexture(c.levelData, c.width, c.height, {
        colorSpace: isSrgb ? 'srgb' : 'linear', ...options,
      });
    }

    if (!this.transcoder) {
      throw new Error('[vela] KTX2 is Basis-supercompressed — call setTranscoder() with a Basis transcoder');
    }
    if (this.transcoder.ready) await this.transcoder.ready;

    // Try to transcode to a GPU-compressed format if the device and transcoder support it.
    const compressedFormat = this.pickCompressedFormat();
    if (compressedFormat && this.transcoder.transcodeToCompressed) {
      const compressed = await this.transcoder.transcodeToCompressed(data, compressedFormat);
      const gpuFormat = isSrgb
        ? (compressedFormat.replace('-unorm', '-unorm-srgb') as string)
        : compressedFormat;
      return new CompressedDataTexture(compressed.blocks, compressed.width, compressed.height, gpuFormat, options);
    }

    // Fallback: transcode to RGBA8.
    const image = await this.transcoder.transcodeToRGBA(data);
    return new DataTexture(image.rgba, image.width, image.height, {
      colorSpace: isSrgb ? 'srgb' : 'linear', ...options,
    });
  }

  /**
   * Pick the best GPU-compressed format the current device supports, in priority
   * order: BC7 → ASTC 4×4 → ETC2. Returns null if none are supported or the
   * device hasn't been registered.
   */
  private pickCompressedFormat(): 'bc7-rgba-unorm' | 'astc-4x4-unorm' | 'etc2-rgba8unorm' | null {
    if (!this.device) return null;
    for (const { format, feature } of COMPRESSED_FORMATS) {
      if (this.device.features.has(feature as GPUFeatureName)) return format;
    }
    return null;
  }
}

/** Parse the KTX2 binary container header + level index (level 0 only). */
function parseContainer(data: Uint8Array): KTX2Container {
  for (let i = 0; i < KTX2_ID.length; i++) {
    if (data[i] !== KTX2_ID[i]) throw new Error('[vela] not a KTX2 file');
  }
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const u32 = (o: number) => dv.getUint32(o, true);

  const vkFormat = u32(12);
  // 16 typeSize, 20 width, 24 height, 28 depth, 32 layerCount, 36 faceCount, 40 levelCount
  const width = u32(20);
  const height = u32(24);
  const levelCount = Math.max(1, u32(40));
  const supercompression = u32(44);

  // Level index starts at byte 80; each entry is 3× uint64 (offset, length, uncompressedLength).
  // Level 0 (largest mip) is the first entry.
  const byteOffset = readU64(dv, 80);
  const byteLength = readU64(dv, 88);
  void levelCount;
  const levelData = data.subarray(byteOffset, byteOffset + byteLength);

  return { vkFormat: vkFormat || VK_UNDEFINED, width, height, supercompression, levelData };
}

/** Read a uint64 as a JS number (KTX2 sizes fit comfortably under 2^53). */
function readU64(dv: DataView, offset: number): number {
  const lo = dv.getUint32(offset, true);
  const hi = dv.getUint32(offset + 4, true);
  return hi * 0x100000000 + lo;
}
