import type { WebGPURenderer, PixelData } from './renderer/WebGPURenderer';
import type { Scene } from './core/Scene';
import type { Camera } from './core/Camera';

/**
 * Golden-image testing helpers: render deterministically, read pixels back,
 * and compare against a reference with actionable failure artifacts. Built
 * for agents — a visual change becomes a runnable assertion.
 *
 * ```ts
 * await expectFrame(renderer, scene, camera, 'golden/lava.png');
 * // throws FrameMismatchError with metrics + actual/diff PNG blobs on failure
 * ```
 */

export interface CaptureOptions {
  /** Frames to render before capturing (TAA/temporal warmup). Default 2. */
  frames?: number;
  /** Deterministic time step per warmup frame. Default 1/60. */
  timeStep?: number;
  /** Starting `renderer.time`. Default 0. */
  startTime?: number;
}

export interface CompareOptions {
  /** Per-channel delta (0..1) below which a pixel counts as equal. Default 0.02. */
  channelTolerance?: number;
  /** Fraction of differing pixels allowed before failing. Default 0.001. */
  maxDiffRatio?: number;
}

export interface CompareResult {
  pass: boolean;
  /** Fraction of pixels with any channel beyond the tolerance. */
  diffRatio: number;
  /** Mean absolute channel delta (0..1). */
  meanDelta: number;
  /** Largest channel delta (0..1). */
  maxDelta: number;
  /** Visual diff: differing pixels in red over a dimmed grayscale base. */
  diff: PixelData;
}

/** Render deterministically and capture the final post-processed canvas frame. */
export async function captureFrame(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: Camera,
  options: CaptureOptions = {},
): Promise<PixelData> {
  const frames = Math.max(1, options.frames ?? 2);
  const timeStep = options.timeStep ?? 1 / 60;
  const prevDeterministic = renderer.deterministic;
  const prevTime = renderer.time;
  renderer.deterministic = true;
  renderer.time = options.startTime ?? 0;
  try {
    for (let i = 0; i < frames - 1; i++) {
      renderer.render(scene, camera);
      renderer.time += timeStep;
    }
    const pixels = renderer.readPixels(); // resolves with the next frame
    renderer.render(scene, camera);
    return await pixels;
  } finally {
    renderer.deterministic = prevDeterministic;
    renderer.time = prevTime;
  }
}

/** Pure pixel comparison (usable offline / in node). */
export function comparePixels(
  actual: PixelData,
  expected: PixelData,
  options: CompareOptions = {},
): CompareResult {
  if (actual.width !== expected.width || actual.height !== expected.height) {
    throw new Error(
      `comparePixels: size mismatch — actual ${actual.width}×${actual.height} vs ` +
      `expected ${expected.width}×${expected.height}. Render at the golden's resolution ` +
      `(renderer.setSize / RenderTarget dimensions) before comparing.`,
    );
  }
  const channelTolerance = options.channelTolerance ?? 0.02;
  const maxDiffRatio = options.maxDiffRatio ?? 0.001;
  const a = actual.data;
  const e = expected.data;
  const n = actual.width * actual.height;
  const diff = new Uint8ClampedArray(n * 4);
  let differing = 0;
  let sumDelta = 0;
  let maxDelta = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    let pixelMax = 0;
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a[o + c] - e[o + c]) / 255;
      sumDelta += d;
      if (d > pixelMax) pixelMax = d;
    }
    if (pixelMax > maxDelta) maxDelta = pixelMax;
    // Dimmed grayscale base; differing pixels flare red by severity.
    const gray = (a[o] + a[o + 1] + a[o + 2]) / 12;
    if (pixelMax > channelTolerance) {
      differing++;
      diff[o] = 255;
      diff[o + 1] = gray;
      diff[o + 2] = gray;
    } else {
      diff[o] = gray; diff[o + 1] = gray; diff[o + 2] = gray;
    }
    diff[o + 3] = 255;
  }
  const diffRatio = differing / n;
  return {
    pass: diffRatio <= maxDiffRatio,
    diffRatio,
    meanDelta: sumDelta / (n * 3),
    maxDelta,
    diff: { data: diff, width: actual.width, height: actual.height },
  };
}

/** Decode a PNG/JPEG URL (or Blob) into pixels. Browser-only. */
export async function loadPixels(source: string | Blob): Promise<PixelData> {
  const blob = typeof source === 'string'
    ? await (await fetch(source)).blob()
    : source;
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return { data: img.data as Uint8ClampedArray<ArrayBuffer>, width: img.width, height: img.height };
}

/** Encode pixels as a PNG Blob. Browser-only. */
export async function pixelsToPNG(pixels: PixelData): Promise<Blob> {
  const canvas = new OffscreenCanvas(pixels.width, pixels.height);
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(new ImageData(pixels.data, pixels.width, pixels.height), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

/** Thrown by `expectFrame` on mismatch; carries inspectable artifacts. */
export class FrameMismatchError extends Error {
  constructor(
    message: string,
    readonly result: CompareResult,
    /** The frame that was actually rendered (save it to update the golden). */
    readonly actual: PixelData,
    /** PNG of the rendered frame. */
    readonly actualPNG: Blob,
    /** PNG visualizing the differences in red. */
    readonly diffPNG: Blob,
  ) {
    super(message);
    this.name = 'FrameMismatchError';
  }
}

/**
 * Render deterministically and assert the frame matches a golden image.
 * On mismatch, throws `FrameMismatchError` with metrics plus the actual and
 * diff frames as PNG blobs (also logged as object URLs for quick viewing).
 */
export async function expectFrame(
  renderer: WebGPURenderer,
  scene: Scene,
  camera: Camera,
  golden: string | Blob | PixelData,
  options: CaptureOptions & CompareOptions = {},
): Promise<CompareResult> {
  const expected = typeof golden === 'string' || golden instanceof Blob
    ? await loadPixels(golden)
    : golden;
  const actual = await captureFrame(renderer, scene, camera, options);
  const result = comparePixels(actual, expected, options);
  if (!result.pass) {
    const actualPNG = await pixelsToPNG(actual);
    const diffPNG = await pixelsToPNG(result.diff);
    const pct = (v: number) => `${(v * 100).toFixed(3)}%`;
    const message =
      `expectFrame: frame differs from golden — ${pct(result.diffRatio)} of pixels ` +
      `beyond tolerance (allowed ${pct(options.maxDiffRatio ?? 0.001)}), ` +
      `mean channel delta ${result.meanDelta.toFixed(4)}, max ${result.maxDelta.toFixed(4)}. ` +
      `Inspect error.actualPNG / error.diffPNG; if the change is intended, save ` +
      `error.actualPNG as the new golden.`;
    console.error(message, {
      actual: URL.createObjectURL(actualPNG),
      diff: URL.createObjectURL(diffPNG),
    });
    throw new FrameMismatchError(message, result, actual, actualPNG, diffPNG);
  }
  return result;
}
