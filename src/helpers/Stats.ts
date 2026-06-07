/**
 * A tiny performance overlay: frames-per-second and frame time (ms), with a
 * rolling bar graph. Call {@link begin} at the start of your frame and {@link end}
 * at the end (or just {@link update} once per frame). The timing math is
 * decoupled from the DOM so it can be verified without a browser.
 */
export class Stats {
  /** The overlay element (null when constructed without a `document`). */
  readonly dom: HTMLDivElement | null = null;

  private beginTime: number;
  private prevTime: number;
  private frames = 0;
  private _fps = 0;
  private _ms = 0;
  private readonly history: number[] = [];
  /** Number of samples kept for the graph. */
  readonly historySize = 80;

  private fpsText: HTMLDivElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(now: number = nowDefault()) {
    this.beginTime = now;
    this.prevTime = now;
    if (typeof document !== 'undefined') this.dom = this.createDom();
  }

  /** Most recent frames-per-second (updated about once a second). */
  get fps(): number {
    return this._fps;
  }
  /** Most recent measured frame time in milliseconds. */
  get ms(): number {
    return this._ms;
  }

  /** Mark the start of a frame. */
  begin(now: number = nowDefault()): void {
    this.beginTime = now;
  }

  /** Mark the end of a frame; returns `now`. Recomputes fps once per second. */
  end(now: number = nowDefault()): number {
    this.frames++;
    this._ms = now - this.beginTime;

    if (now >= this.prevTime + 1000) {
      this._fps = Math.round((this.frames * 1000) / (now - this.prevTime));
      this.prevTime = now;
      this.frames = 0;
      this.pushHistory(this._fps);
      this.render();
    }
    return now;
  }

  /** Convenience: `end()` then `begin()` for the next frame. */
  update(now: number = nowDefault()): void {
    this.begin(this.end(now));
  }

  private pushHistory(fps: number): void {
    this.history.push(fps);
    if (this.history.length > this.historySize) this.history.shift();
  }

  /** Snapshot of recent per-second fps samples (oldest first). */
  getHistory(): readonly number[] {
    return this.history;
  }

  private createDom(): HTMLDivElement {
    const dom = document.createElement('div');
    dom.style.cssText =
      'position:fixed;top:0;left:0;z-index:10000;padding:4px 6px;' +
      'font:11px/1.4 monospace;color:#0ff;background:rgba(0,0,0,0.6);user-select:none';

    this.fpsText = document.createElement('div');
    this.fpsText.textContent = '-- fps';
    dom.appendChild(this.fpsText);

    this.canvas = document.createElement('canvas');
    this.canvas.width = this.historySize;
    this.canvas.height = 32;
    this.canvas.style.cssText = 'display:block;width:100%;height:32px;margin-top:2px';
    this.ctx = this.canvas.getContext('2d');
    dom.appendChild(this.canvas);
    return dom;
  }

  private render(): void {
    if (this.fpsText) this.fpsText.textContent = `${this._fps} fps  ${this._ms.toFixed(1)} ms`;
    const ctx = this.ctx;
    const canvas = this.canvas;
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const max = Math.max(60, ...this.history);
    ctx.fillStyle = '#0ff';
    for (let i = 0; i < this.history.length; i++) {
      const barH = Math.max(1, Math.round((this.history[i] / max) * h));
      ctx.fillRect(w - this.history.length + i, h - barH, 1, barH);
    }
  }
}

function nowDefault(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}
