import { DataTexture } from '../textures/DataTexture';
import type { TextureOptions } from '../textures/Texture';

/**
 * Loads Radiance `.hdr` (RGBE) equirectangular environment maps into a float
 * {@link DataTexture}, suitable for `scene.environment`. Supports the new-format
 * RLE scanlines and the flat/old fallback.
 */
export class RGBELoader {
  async load(url: string, options?: TextureOptions): Promise<DataTexture> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`[vela] failed to load HDR: ${url} (${res.status})`);
    return this.parse(await res.arrayBuffer(), options);
  }

  /** Parse `.hdr` bytes into an RGBA Float32 {@link DataTexture}. */
  parse(buffer: ArrayBuffer, options: TextureOptions = {}): DataTexture {
    const bytes = new Uint8Array(buffer);
    let pos = 0;

    // --- header: ASCII lines terminated by a blank line ---
    const readLine = (): string => {
      let s = '';
      while (pos < bytes.length && bytes[pos] !== 0x0a) s += String.fromCharCode(bytes[pos++]);
      pos++; // skip newline
      return s;
    };
    const magic = readLine();
    if (!magic.startsWith('#?')) throw new Error('[vela] not a Radiance HDR file');
    let line = readLine();
    while (line !== '') line = readLine(); // skip FORMAT/EXPOSURE/etc until blank

    // resolution: "-Y H +X W"
    const res = readLine().match(/-Y (\d+) \+X (\d+)/);
    if (!res) throw new Error('[vela] unsupported HDR resolution line');
    const height = parseInt(res[1], 10);
    const width = parseInt(res[2], 10);

    const rgbe = new Uint8Array(width * height * 4);
    const useRLE = width >= 8 && width < 32768;

    for (let y = 0; y < height; y++) {
      const rowOffset = y * width * 4;
      if (useRLE && bytes[pos] === 2 && bytes[pos + 1] === 2 &&
          ((bytes[pos + 2] << 8) | bytes[pos + 3]) === width) {
        pos += 4;
        // Four separate RLE-compressed channels.
        for (let c = 0; c < 4; c++) {
          let x = 0;
          while (x < width) {
            let count = bytes[pos++];
            if (count > 128) {
              // run of (count-128) identical bytes
              count -= 128;
              const value = bytes[pos++];
              for (let i = 0; i < count; i++) rgbe[rowOffset + (x++) * 4 + c] = value;
            } else {
              // literal run of `count` bytes
              for (let i = 0; i < count; i++) rgbe[rowOffset + (x++) * 4 + c] = bytes[pos++];
            }
          }
        }
      } else {
        // Flat scanline: width * RGBE quadruples.
        for (let x = 0; x < width * 4; x++) rgbe[rowOffset + x] = bytes[pos++];
      }
    }

    // RGBE -> linear float RGBA (alpha = 1).
    const data = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      const e = rgbe[i * 4 + 3];
      if (e > 0) {
        const f = Math.pow(2, e - 136); // 2^(e - 128 - 8)
        data[i * 4] = rgbe[i * 4] * f;
        data[i * 4 + 1] = rgbe[i * 4 + 1] * f;
        data[i * 4 + 2] = rgbe[i * 4 + 2] * f;
      }
      data[i * 4 + 3] = 1;
    }

    return new DataTexture(data, width, height, { wrapS: 'repeat', wrapT: 'clamp', ...options });
  }
}
