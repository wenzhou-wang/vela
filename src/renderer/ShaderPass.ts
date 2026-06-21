import type { UniformValue } from '../materials/ShaderMaterial';

/** Optional scene buffers a ShaderPass can request. */
export type ShaderPassInput = 'normal' | 'linearDepth';

export interface ShaderPassOptions {
  /**
   * WGSL defining `fn effect(uv : vec2<f32>) -> vec4<f32>`, returning the
   * output color for that pixel. Available in scope:
   * - `sceneColor(uv)` / `sceneTex` + `sceneSmp` — the previous stage (HDR linear).
   * - `sceneDepth(uv)` — non-linear depth in [0,1] (1 = background).
   * - `sceneWorldNormal(uv)` / `sceneViewNormal(uv)` — when `inputs` requests `normal`.
   * - `sceneLinearDepth(uv)` — view-space distance when `inputs` requests `linearDepth`.
   * - `pp.resolution` (xy = pixels, zw = 1/pixels), `pp.time.x` (seconds).
   * - `u.<name>` scalars and `t_<name>`/`s_<name>` textures from `uniforms`.
   */
  effect: string;
  /** Custom uniforms (same types as ShaderMaterial): `u.<name>` / `t_<name>`. */
  uniforms?: Record<string, UniformValue>;
  /** Opt into additional scene buffers used by this pass. */
  inputs?: ShaderPassInput[];
  /** Skip this pass without removing it from the chain. */
  enabled?: boolean;
  name?: string;
}

/**
 * A custom fullscreen post-processing effect. Push instances into
 * `renderer.passes`; they run in order in HDR linear space after the scene
 * (and bloom/SSAO/TAA) and before tonemap, each reading the previous stage's
 * color plus the scene depth.
 *
 * ```ts
 * import { ShaderPass } from 'vela';
 * renderer.postProcessing = true;
 * renderer.passes.push(new ShaderPass({
 *   effect: `
 *     fn effect(uv : vec2<f32>) -> vec4<f32> {
 *       let off = vec2(sin(uv.y * 80.0 + pp.time.x * 4.0) * u.amount, 0.0);
 *       return sceneColor(uv + off);
 *     }
 *   `,
 *   uniforms: { amount: 0.01 },
 * }));
 * ```
 */
export class ShaderPass {
  readonly isShaderPass = true;
  enabled: boolean;
  name: string;
  uniforms: Record<string, UniformValue>;
  /** Additional scene buffers this pass needs. Mutate freely between frames. */
  inputs: ShaderPassInput[];
  /** Bumped by `setEffect()`; part of the pipeline cache key. */
  version = 0;
  /** Stable id for caching GPU resources. */
  readonly id: string;
  private static _nextId = 0;
  private _effect: string;

  constructor(options: ShaderPassOptions) {
    ShaderPass.validate(options.effect);
    this.id = `shaderpass-${ShaderPass._nextId++}`;
    this._effect = options.effect;
    this.uniforms = options.uniforms ?? {};
    this.inputs = options.inputs ?? [];
    this.enabled = options.enabled ?? true;
    this.name = options.name ?? 'ShaderPass';
  }

  get effectCode(): string {
    return this._effect;
  }

  /** Replace the effect function; recompiles on the next frame. */
  setEffect(code: string): void {
    ShaderPass.validate(code);
    this._effect = code;
    this.version++;
  }

  private static validate(code: string): void {
    if (typeof code !== 'string' || !/fn\s+effect\s*\(/.test(code)) {
      throw new Error(
        'ShaderPass: the `effect` option must be WGSL source defining\n' +
        '  fn effect(uv : vec2<f32>) -> vec4<f32> { ... }\n' +
        'Sample the previous stage with sceneColor(uv); return the output color.',
      );
    }
  }
}
