function srgbToLinear(c: number): number {
  return c < 0.04045 ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4);
}

function linearToSRGB(c: number): number {
  return c < 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 0.41666) - 0.055;
}

/**
 * Anything accepted where a color is expected: a {@link Color}, or a linear
 * `[r, g, b]` (optionally `[r, g, b, a]`) array. Packed `0xRRGGBB` hex is
 * deliberately not accepted — use `new Color().setHex(...)` to opt into sRGB hex.
 */
export type ColorInput = Color | readonly number[];

/** A linear-space RGB color. Hex/CSS inputs are assumed sRGB and converted. */
export class Color {
  r: number;
  g: number;
  b: number;

  constructor(r = 1, g = 1, b = 1) {
    this.r = r;
    this.g = g;
    this.b = b;
  }

  set(r: number, g: number, b: number): this {
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }

  copy(c: Color): this {
    this.r = c.r;
    this.g = c.g;
    this.b = c.b;
    return this;
  }

  clone(): Color {
    return new Color(this.r, this.g, this.b);
  }

  /** Set from a packed 0xRRGGBB integer (sRGB), converting to linear. */
  setHex(hex: number): this {
    this.r = srgbToLinear(((hex >> 16) & 255) / 255);
    this.g = srgbToLinear(((hex >> 8) & 255) / 255);
    this.b = srgbToLinear((hex & 255) / 255);
    return this;
  }

  /** Copy from a {@link Color}, or read linear channels from an `[r, g, b]` array. */
  setFrom(value: ColorInput): this {
    if (value instanceof Color) return this.copy(value);
    return this.fromArray(value);
  }

  /** Set directly in linear space (no conversion). */
  setRGB(r: number, g: number, b: number): this {
    this.r = r;
    this.g = g;
    this.b = b;
    return this;
  }

  getHex(): number {
    const r = Math.round(linearToSRGB(this.r) * 255);
    const g = Math.round(linearToSRGB(this.g) * 255);
    const b = Math.round(linearToSRGB(this.b) * 255);
    return (r << 16) ^ (g << 8) ^ b;
  }

  multiplyScalar(s: number): this {
    this.r *= s;
    this.g *= s;
    this.b *= s;
    return this;
  }

  fromArray(a: ArrayLike<number>, offset = 0): this {
    this.r = a[offset];
    this.g = a[offset + 1];
    this.b = a[offset + 2];
    return this;
  }
}
