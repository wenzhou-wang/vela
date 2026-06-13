import { Texture } from '../textures/Texture';

/**
 * An offscreen render destination: `renderer.render(scene, camera, rt)` draws
 * into it, and `rt.texture` works anywhere a sampled texture does — mirrors,
 * portals, minimaps, security monitors:
 *
 * ```ts
 * const rt = new RenderTarget(512, 512);
 * monitor.material = new StandardMaterial({ map: rt.texture });
 * // each frame:
 * renderer.render(scene, securityCamera, rt);
 * renderer.render(scene, mainCamera);
 * ```
 *
 * Render-target passes use the direct pipeline (in-shader tonemapping); the
 * post-processing chain applies only to the default (canvas) target.
 * Remember to set the camera's aspect to `width / height`.
 */
export class RenderTarget {
  readonly isRenderTarget = true;
  readonly width: number;
  readonly height: number;
  /** Sample this in any material (`map`, sprite texture, ...). */
  readonly texture: Texture;

  constructor(width: number, height: number) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
      throw new Error(`RenderTarget: width/height must be positive integers, got ${width}x${height}.`);
    }
    this.width = width;
    this.height = height;
    this.texture = new Texture(null, { generateMipmaps: false, colorSpace: 'srgb' });
  }
}
