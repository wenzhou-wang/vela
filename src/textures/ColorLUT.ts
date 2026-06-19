/**
 * A 3-D color lookup table for display-space color grading, applied as the final
 * post pass (after tonemap). `data` is `size³ × 3` RGB, red varying fastest —
 * the layout of a `.cube` LUT and of a WebGPU 3-D texture (x = R, y = G, z = B).
 */
export class ColorLUT {
  readonly size: number;
  readonly data: Float32Array;
  /** Input domain from `.cube` DOMAIN_MIN/MAX, per channel (default [0, 1]). */
  domainMin: [number, number, number] = [0, 0, 0];
  domainMax: [number, number, number] = [1, 1, 1];
  /** Blend toward the graded color in [0, 1] (1 = full grade). */
  strength = 1;
  /** Bumped to force the post pipeline to re-upload the 3-D texture. */
  version = 0;

  constructor(size: number, data: Float32Array, strength = 1) {
    if (data.length !== size * size * size * 3) {
      throw new Error(`ColorLUT: data length ${data.length} != size^3*3 (${size * size * size * 3})`);
    }
    this.size = size;
    this.data = data;
    this.strength = strength;
  }

  /**
   * Parse an Adobe/Resolve `.cube` 3-D LUT. Honors `LUT_3D_SIZE` and an optional
   * `DOMAIN_MIN`/`DOMAIN_MAX` (the input domain — applied to the lookup coordinate
   * at sample time; table entries are the output colors, stored verbatim).
   * Throws on a 1-D LUT or a row-count mismatch.
   */
  static parseCube(text: string): ColorLUT {
    let size = 0;
    const domainMin = [0, 0, 0];
    const domainMax = [1, 1, 1];
    const values: number[] = [];

    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (line.length === 0 || line[0] === '#') continue;
      const upper = line.toUpperCase();
      if (upper.startsWith('LUT_3D_SIZE')) {
        size = parseInt(line.split(/\s+/)[1], 10);
      } else if (upper.startsWith('LUT_1D_SIZE')) {
        throw new Error('ColorLUT: 1-D .cube LUTs are not supported');
      } else if (upper.startsWith('DOMAIN_MIN')) {
        const p = line.split(/\s+/); domainMin[0] = +p[1]; domainMin[1] = +p[2]; domainMin[2] = +p[3];
      } else if (upper.startsWith('DOMAIN_MAX')) {
        const p = line.split(/\s+/); domainMax[0] = +p[1]; domainMax[1] = +p[2]; domainMax[2] = +p[3];
      } else if (upper.startsWith('TITLE')) {
        continue;
      } else {
        const p = line.split(/\s+/);
        if (p.length >= 3) { values.push(+p[0], +p[1], +p[2]); }
      }
    }

    if (size <= 0) throw new Error('ColorLUT: missing LUT_3D_SIZE');
    const expected = size * size * size * 3;
    if (values.length !== expected) {
      throw new Error(`ColorLUT: ${values.length / 3} entries, expected ${size * size * size}`);
    }

    // Table entries are the output colors — stored verbatim. DOMAIN_MIN/MAX
    // describe the input domain and are honored at sample time (they remap the
    // lookup coordinate), not by scaling the outputs.
    const lut = new ColorLUT(size, new Float32Array(values));
    lut.domainMin = [domainMin[0], domainMin[1], domainMin[2]];
    lut.domainMax = [domainMax[0], domainMax[1], domainMax[2]];
    return lut;
  }
}
