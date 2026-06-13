import { Object3D } from './Object3D';
import { Color } from '../math/Color';

export interface TextMeshOptions {
  /** CSS font family/weight, e.g. `'bold sans-serif'`. Default `'sans-serif'`. */
  font?: string;
  /** Em height: world units, or CSS pixels when `screenSpace`. Default 0.5. */
  fontSize?: number;
  color?: Color;
  opacity?: number;
  /** Draw as a HUD overlay (no depth test, sized in CSS pixels). */
  screenSpace?: boolean;
  /** Horizontal alignment around the object position. Default 'center'. */
  anchor?: 'left' | 'center' | 'right';
}

/**
 * Billboarded SDF text. Glyphs rasterize lazily into a per-font distance-field
 * atlas and render through the sprite batcher — one instanced draw per font —
 * so text stays crisp at any scale and cheap at any count. Supports `\n` for
 * multiple lines. Mutate `text` (or any option) freely; everything re-reads
 * each frame.
 *
 * ```ts
 * const label = new TextMesh('Player 1', { fontSize: 0.3 });
 * label.position.set(0, 2, 0);
 * scene.add(label);
 * ```
 */
export class TextMesh extends Object3D {
  readonly isTextMesh = true;
  text: string;
  font: string;
  fontSize: number;
  color: Color;
  opacity: number;
  screenSpace: boolean;
  anchor: 'left' | 'center' | 'right';

  constructor(text = '', options: TextMeshOptions = {}) {
    super();
    this.type = 'TextMesh';
    this.text = text;
    this.font = options.font ?? 'sans-serif';
    this.fontSize = options.fontSize ?? 0.5;
    this.color = options.color ?? new Color(1, 1, 1);
    this.opacity = options.opacity ?? 1;
    this.screenSpace = options.screenSpace ?? false;
    this.anchor = options.anchor ?? 'center';
  }
}
