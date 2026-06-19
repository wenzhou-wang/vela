import { DataTexture } from './DataTexture';
import { Color, type ColorInput } from '../math/Color';

/** One color stop along a gradient: a `position` in [0, 1] with a linear `color`. */
export interface GradientStop {
  position: number;
  color: ColorInput;
  /** Optional alpha in [0, 1] (default 1). */
  opacity?: number;
}

const _ca = new Color(), _cb = new Color();

/**
 * Build a 1-D gradient as a `width × 1` linear `DataTexture` (clamp-wrapped,
 * linearly filtered) — the generic primitive behind toon ramps, gradient maps,
 * and color LUTs. Stops take `ColorInput` (a `Color` or linear `[r, g, b]`), so
 * there is no hidden color space. Plug it into a `ShaderMaterial`/`ShaderPass`
 * texture uniform and sample it at `(t, 0.5)`:
 *
 * ```wgsl
 * let ramp = textureSample(t_ramp, s_ramp, vec2<f32>(ndotl, 0.5)).rgb;
 * ```
 */
export function gradientTexture(stops: GradientStop[], width = 256): DataTexture {
  if (stops.length === 0) throw new Error('gradientTexture: at least one stop is required');
  const sorted = [...stops].sort((a, b) => a.position - b.position);
  const data = new Float32Array(width * 4);

  for (let x = 0; x < width; x++) {
    const t = width > 1 ? x / (width - 1) : 0;
    // Find the bracketing stops (clamp at the ends).
    let i = 0;
    while (i < sorted.length - 1 && sorted[i + 1].position <= t) i++;
    const s0 = sorted[i];
    const s1 = sorted[Math.min(i + 1, sorted.length - 1)];
    const span = s1.position - s0.position;
    const a = span > 1e-6 ? Math.max(0, Math.min(1, (t - s0.position) / span)) : 0;

    _ca.setFrom(s0.color);
    _cb.setFrom(s1.color);
    const o = x * 4;
    data[o] = _ca.r + (_cb.r - _ca.r) * a;
    data[o + 1] = _ca.g + (_cb.g - _ca.g) * a;
    data[o + 2] = _ca.b + (_cb.b - _ca.b) * a;
    const a0 = s0.opacity ?? 1, a1 = s1.opacity ?? 1;
    data[o + 3] = a0 + (a1 - a0) * a;
  }

  return new DataTexture(data, width, 1, {
    wrapS: 'clamp', wrapT: 'clamp',
    magFilter: 'linear', minFilter: 'linear',
    generateMipmaps: false, colorSpace: 'linear',
  });
}
