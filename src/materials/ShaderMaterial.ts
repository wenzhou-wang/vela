import { Material, Side } from './Material';
import { Color } from '../math/Color';
import { Vector2 } from '../math/Vector2';
import { Vector3 } from '../math/Vector3';
import { Vector4 } from '../math/Vector4';

/** Value types accepted in `ShaderMaterial.uniforms`. */
export type UniformValue = number | Vector2 | Vector3 | Vector4 | Color;

type UniformKind = 'f32' | 'vec2' | 'vec3' | 'vec4';

export interface UniformField {
  name: string;
  kind: UniformKind;
  /** Offset into the packed buffer, in floats. */
  offset: number;
}

export interface UniformLayout {
  /** WGSL `struct SMUniforms {...}` + `@group(2) @binding(0)` declaration ('' when no uniforms). */
  wgsl: string;
  /** Packed buffer size in bytes (multiple of 16, min 16). */
  size: number;
  fields: UniformField[];
}

const WGSL_TYPE: Record<UniformKind, string> = {
  f32: 'f32',
  vec2: 'vec2<f32>',
  vec3: 'vec3<f32>',
  vec4: 'vec4<f32>',
};
// Alignment and size in floats per WGSL uniform layout rules.
const ALIGN: Record<UniformKind, number> = { f32: 1, vec2: 2, vec3: 4, vec4: 4 };
const SIZE: Record<UniformKind, number> = { f32: 1, vec2: 2, vec3: 3, vec4: 4 };

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

function kindOf(name: string, value: UniformValue): UniformKind {
  if (typeof value === 'number') return 'f32';
  if (value instanceof Vector2) return 'vec2';
  if (value instanceof Vector3 || value instanceof Color) return 'vec3';
  if (value instanceof Vector4) return 'vec4';
  throw new Error(
    `ShaderMaterial uniform "${name}" has an unsupported value. ` +
    `Use a number (f32), Vector2 (vec2<f32>), Vector3 or Color (vec3<f32>), or Vector4 (vec4<f32>).`,
  );
}

/**
 * Compute the WGSL struct + packed layout for a uniforms object. Fields follow
 * declaration order with WGSL alignment rules (vec3 aligns to 16 bytes; an f32
 * may fill its padding). Pure — exported for offline verification.
 */
export function computeUniformLayout(uniforms: Record<string, UniformValue>): UniformLayout {
  const names = Object.keys(uniforms);
  if (names.length === 0) return { wgsl: '', size: 16, fields: [] };

  const fields: UniformField[] = [];
  let offset = 0; // in floats
  let lines = 'struct SMUniforms {\n';
  for (const name of names) {
    if (!IDENT.test(name)) {
      throw new Error(
        `ShaderMaterial uniform "${name}" is not a valid WGSL identifier ` +
        `(letters, digits, underscore; must not start with a digit).`,
      );
    }
    const kind = kindOf(name, uniforms[name]);
    offset = Math.ceil(offset / ALIGN[kind]) * ALIGN[kind];
    fields.push({ name, kind, offset });
    lines += `  ${name} : ${WGSL_TYPE[kind]},\n`;
    offset += SIZE[kind];
  }
  lines += '};\n@group(2) @binding(0) var<uniform> u : SMUniforms;\n';
  const size = Math.max(16, Math.ceil(offset / 4) * 16);
  return { wgsl: lines, size, fields };
}

/** Pack current uniform values into `out` according to `layout`. */
export function packUniforms(
  uniforms: Record<string, UniformValue>,
  layout: UniformLayout,
  out: Float32Array,
): void {
  for (const f of layout.fields) {
    const v = uniforms[f.name];
    const o = f.offset;
    if (typeof v === 'number') {
      out[o] = v;
    } else if (v instanceof Color) {
      out[o] = v.r; out[o + 1] = v.g; out[o + 2] = v.b;
    } else if (v instanceof Vector2) {
      out[o] = v.x; out[o + 1] = v.y;
    } else if (v instanceof Vector3) {
      out[o] = v.x; out[o + 1] = v.y; out[o + 2] = v.z;
    } else if (v instanceof Vector4) {
      out[o] = v.x; out[o + 1] = v.y; out[o + 2] = v.z; out[o + 3] = v.w;
    }
  }
}

export interface ShaderMaterialOptions {
  /**
   * WGSL source defining `fn surface(in : VSOut) -> Surface`. Start from
   * `defaultSurface(in)` and override fields; custom uniforms are available as
   * `u.<name>`, frame data as `frame.*`, and `elapsedTime()` returns seconds.
   */
  surface: string;
  /** Custom uniforms exposed to WGSL as `u.<name>`; values are re-uploaded every frame. */
  uniforms?: Record<string, UniformValue>;
  name?: string;
  side?: Side;
  transparent?: boolean;
  opacity?: number;
  depthTest?: boolean;
  depthWrite?: boolean;
}

/**
 * A custom-surface material: you write one WGSL function that produces PBR
 * surface inputs (albedo, roughness, emissive, normal, ...) and the engine
 * supplies everything else — all light types, shadows, IBL, clustered
 * lighting, OIT, post-processing and tonemapping behave exactly as they do
 * for StandardMaterial.
 *
 * ```ts
 * const mat = new ShaderMaterial({
 *   surface: `
 *     fn surface(in : VSOut) -> Surface {
 *       var s = defaultSurface(in);
 *       s.baseColor = vec3(in.uv, 0.5 + 0.5 * sin(elapsedTime()));
 *       s.roughness = u.gloss;
 *       return s;
 *     }
 *   `,
 *   uniforms: { gloss: 0.2 },
 * });
 * mat.uniforms.gloss = 0.8; // updates next frame, no flags needed
 * ```
 */
export class ShaderMaterial extends Material {
  readonly isShaderMaterial = true;
  readonly type = 'ShaderMaterial';

  /** Custom uniform values; mutate freely (uploaded every frame). Adding/removing keys recompiles. */
  uniforms: Record<string, UniformValue>;
  /** Bumped by setSurface(); part of the pipeline cache key. */
  version = 0;

  private _surfaceCode: string;

  constructor(options: ShaderMaterialOptions) {
    super();
    ShaderMaterial.validateSurface(options.surface);
    this._surfaceCode = options.surface;
    this.uniforms = options.uniforms ?? {};
    if (options.name !== undefined) this.name = options.name;
    if (options.side !== undefined) this.side = options.side;
    if (options.transparent !== undefined) this.transparent = options.transparent;
    if (options.opacity !== undefined) this.opacity = options.opacity;
    if (options.depthTest !== undefined) this.depthTest = options.depthTest;
    if (options.depthWrite !== undefined) this.depthWrite = options.depthWrite;
  }

  get surfaceCode(): string {
    return this._surfaceCode;
  }

  /** Replace the surface function; the shader recompiles on next draw. */
  setSurface(code: string): void {
    ShaderMaterial.validateSurface(code);
    this._surfaceCode = code;
    this.version++;
  }

  private static validateSurface(code: string): void {
    if (typeof code !== 'string' || !/fn\s+surface\s*\(/.test(code)) {
      throw new Error(
        'ShaderMaterial: the `surface` option must be WGSL source defining\n' +
        '  fn surface(in : VSOut) -> Surface { ... }\n' +
        'Tip: start with `var s = defaultSurface(in);`, set fields like s.baseColor / ' +
        's.roughness / s.emissive / s.normal, and `return s;`.',
      );
    }
  }
}
