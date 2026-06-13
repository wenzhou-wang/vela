import { DataTexture } from './DataTexture';

/** Placement + metrics for one glyph, in pixels at BASE_FONT size. */
export interface GlyphInfo {
  /** Atlas uv rect (top-left / bottom-right). */
  u0: number; v0: number; u1: number; v1: number;
  /** Quad size (glyph box expanded by the SDF spread). */
  w: number; h: number;
  /** Quad-center offset from the pen position on the baseline (+y up). */
  cx: number; cy: number;
  /** Horizontal pen advance. */
  advance: number;
}

const ATLAS_SIZE = 1024;
const CELL = 64;
const COLS = ATLAS_SIZE / CELL; // 16 → 256 glyph cells
const BASE_FONT = 32;           // raster size in px; quads scale from this
const SPREAD = 6;               // SDF half-width in px
const DRAW_X = 14;              // pen origin inside a cell
const BASELINE_Y = 44;

export { BASE_FONT as SDF_BASE_FONT };

/**
 * A lazily-populated signed-distance-field glyph atlas for one font. Glyphs
 * rasterize on demand via Canvas2D, get a Euclidean distance transform
 * (Felzenszwalb two-pass), and land in a 64px-cell grid inside one
 * 1024×1024 rgba8 texture (distance in alpha). `texture.version` bumps on
 * every new glyph so the renderer re-uploads automatically.
 */
export class SDFFontAtlas {
  readonly texture: DataTexture;
  /** Line advance in px at BASE_FONT. */
  readonly lineHeight = Math.round(BASE_FONT * 1.25);

  private glyphs = new Map<string, GlyphInfo | null>();
  private nextCell = 0;
  private data: Uint8Array;
  private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  private warned = false;

  constructor(readonly font: string) {
    this.data = new Uint8Array(ATLAS_SIZE * ATLAS_SIZE * 4);
    this.texture = new DataTexture(this.data, ATLAS_SIZE, ATLAS_SIZE, {
      generateMipmaps: false,
      colorSpace: 'linear',
    });
    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(CELL, CELL)
      : (() => { const c = document.createElement('canvas'); c.width = c.height = CELL; return c; })();
    const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
      OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
    if (!ctx) throw new Error('SDFFontAtlas: Canvas2D is unavailable (needed to rasterize glyphs).');
    this.ctx = ctx;
    ctx.font = `${BASE_FONT}px ${font}`;
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#fff';
  }

  /** Get (rasterizing if needed) a glyph; null for whitespace/unfittable glyphs. */
  getGlyph(ch: string): GlyphInfo | null {
    let info = this.glyphs.get(ch);
    if (info !== undefined) return info;
    info = this.rasterize(ch);
    this.glyphs.set(ch, info);
    return info;
  }

  /** Pen advance for a character (works for whitespace too). */
  advanceOf(ch: string): number {
    const g = this.getGlyph(ch);
    if (g) return g.advance;
    return this.ctx.measureText(ch).width;
  }

  private rasterize(ch: string): GlyphInfo | null {
    const m = this.ctx.measureText(ch);
    const left = Math.ceil(m.actualBoundingBoxLeft ?? 0);
    const right = Math.ceil(m.actualBoundingBoxRight ?? 0);
    const ascent = Math.ceil(m.actualBoundingBoxAscent ?? 0);
    const descent = Math.ceil(m.actualBoundingBoxDescent ?? 0);
    if (right + left <= 0 || ascent + descent <= 0) return null; // whitespace

    if (this.nextCell >= COLS * COLS) {
      if (!this.warned) {
        this.warned = true;
        console.warn(
          `[vela] SDFFontAtlas("${this.font}") is full (${COLS * COLS} glyphs); ` +
          'further unique characters will not render. Split text across fonts to fix.',
        );
      }
      return null;
    }

    // Rasterize into the scratch cell.
    this.ctx.clearRect(0, 0, CELL, CELL);
    this.ctx.fillText(ch, DRAW_X, BASELINE_Y);
    const img = this.ctx.getImageData(0, 0, CELL, CELL).data;

    // Euclidean distance transform of coverage and its complement.
    const n = CELL * CELL;
    const inside = new Float32Array(n);
    const outside = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const covered = img[i * 4 + 3] >= 128;
      inside[i] = covered ? INF : 0;
      outside[i] = covered ? 0 : INF;
    }
    edt2d(inside, CELL, CELL);
    edt2d(outside, CELL, CELL);

    // Pack the distance field into the cell's atlas region (alpha channel;
    // rgb stays white so tinting works).
    const cell = this.nextCell++;
    const cellX = (cell % COLS) * CELL;
    const cellY = Math.floor(cell / COLS) * CELL;
    for (let y = 0; y < CELL; y++) {
      for (let x = 0; x < CELL; x++) {
        const i = y * CELL + x;
        const signed = Math.sqrt(outside[i]) - Math.sqrt(inside[i]); // + outside
        const a = Math.max(0, Math.min(255, Math.round((0.5 - signed / (2 * SPREAD)) * 255)));
        const o = ((cellY + y) * ATLAS_SIZE + cellX + x) * 4;
        this.data[o] = 255; this.data[o + 1] = 255; this.data[o + 2] = 255; this.data[o + 3] = a;
      }
    }
    this.texture.needsUpdate();

    // Quad = glyph box expanded by the spread, relative to the pen origin.
    const x0 = DRAW_X - left - SPREAD;
    const x1 = DRAW_X + right + SPREAD;
    const y0 = BASELINE_Y - ascent - SPREAD;  // top, canvas coords (+y down)
    const y1 = BASELINE_Y + descent + SPREAD;
    return {
      u0: (cellX + x0) / ATLAS_SIZE,
      v0: (cellY + y0) / ATLAS_SIZE,
      u1: (cellX + x1) / ATLAS_SIZE,
      v1: (cellY + y1) / ATLAS_SIZE,
      w: x1 - x0,
      h: y1 - y0,
      cx: (x0 + x1) / 2 - DRAW_X,
      cy: BASELINE_Y - (y0 + y1) / 2, // +y up
      advance: m.width,
    };
  }
}

const INF = 1e20;

// Felzenszwalb & Huttenlocher 1D squared-distance transform.
function edt1d(f: Float32Array, d: Float32Array, v: Int32Array, z: Float32Array, n: number): void {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

/** In-place 2D squared Euclidean distance transform (exported for offline tests). */
export function edt2d(grid: Float32Array, w: number, h: number): void {
  const size = Math.max(w, h);
  const f = new Float32Array(size);
  const d = new Float32Array(size);
  const v = new Int32Array(size);
  const z = new Float32Array(size + 1);
  for (let x = 0; x < w; x++) {           // columns
    for (let y = 0; y < h; y++) f[y] = grid[y * w + x];
    edt1d(f, d, v, z, h);
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {           // rows
    for (let x = 0; x < w; x++) f[x] = grid[y * w + x];
    edt1d(f, d, v, z, w);
    for (let x = 0; x < w; x++) grid[y * w + x] = d[x];
  }
}
