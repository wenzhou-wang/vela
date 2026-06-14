/**
 * Comic / cel look — implemented entirely in application code on top of the
 * generic engine APIs (ShaderMaterial + ShaderPass). The engine itself knows
 * nothing about cel shading; this is the reference for how a use-case shader
 * lives in the app, per docs/ARCHITECTURE.md ("the engine stays generic").
 *
 * Two parts:
 *  1. Flat albedo — each mesh's StandardMaterial is swapped for a ShaderMaterial
 *     whose surface emits the authored base color as unlit emissive (no lighting,
 *     no specular), giving the flat comic fill. Originals are restored on exit.
 *  2. Ink outlines — a fullscreen ShaderPass detects silhouette (depth), interior
 *     creases (reconstructed view normals), and texture color edges, and draws
 *     dark ink over them.
 */
import {
  Object3D, Mesh, StandardMaterial, ShaderMaterial, ShaderPass, Texture, Color,
} from 'vela';

// Flat unlit albedo: no lit diffuse/specular; emissive carries the base color.
const FLAT_SURFACE = /* wgsl */ `
  fn surface(in : VSOut) -> Surface {
    var s = defaultSurface(in);
    s.baseColor = vec3<f32>(0.0);
    s.metalness = 0.0;
    s.roughness = 1.0;
    s.emissive = textureSample(t_albedo, s_albedo, in.uv).rgb * u.tint;
    return s;
  }
`;

// Outline effect: silhouette + normal crease + color edge → ink over the scene.
const OUTLINE_EFFECT = /* wgsl */ `
  fn isGeo(uv : vec2<f32>) -> f32 { return select(0.0, 1.0, sceneDepth(uv) < 1.0); }

  fn effect(uv : vec2<f32>) -> vec4<f32> {
    let c = sceneColor(uv);
    let step = pp.resolution.zw * u.thickness;
    let dC = sceneDepth(uv);
    let object = isGeo(uv);

    // Silhouette: depth occupancy changing across an 8-neighborhood.
    var outer = 0.0;
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>(-step.x, 0.0))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>( step.x, 0.0))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>(0.0, -step.y))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>(0.0,  step.y))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>(-step.x, -step.y))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>( step.x,  step.y))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>(-step.x,  step.y))));
    outer = max(outer, abs(object - isGeo(uv + vec2<f32>( step.x, -step.y))));

    // Interior creases: angle between view-space normals reconstructed from depth.
    let inner = pp.resolution.zw;
    let uvL = uv - vec2<f32>(inner.x, 0.0); let uvR = uv + vec2<f32>(inner.x, 0.0);
    let uvU = uv - vec2<f32>(0.0, inner.y); let uvD = uv + vec2<f32>(0.0, inner.y);
    let pC = viewPosition(uv, dC);
    let pL = viewPosition(uvL, sceneDepth(uvL)); let pR = viewPosition(uvR, sceneDepth(uvR));
    let pU = viewPosition(uvU, sceneDepth(uvU)); let pD = viewPosition(uvD, sceneDepth(uvD));
    let n = normalize(cross(pR - pL, pD - pU));
    let nL = normalize(cross(pC - pL, viewPosition(uvL - vec2<f32>(0.0, inner.y), sceneDepth(uvL - vec2<f32>(0.0, inner.y))) - pL));
    let crease = smoothstep(0.25, 0.6, 1.0 - abs(dot(n, nL))) * object;

    // Color edges (Sobel) from the authored albedo.
    let cL = sceneColor(uvL).rgb; let cR = sceneColor(uvR).rgb;
    let cU = sceneColor(uvU).rgb; let cD = sceneColor(uvD).rgb;
    let colorDelta = (length(cR - cL) + length(cD - cU)) * 0.5;
    let colorEdge = smoothstep(u.colorEdge, u.colorEdge * 2.5, colorDelta) * object;

    let edge = clamp(max(outer, max(crease * 0.8, colorEdge * 0.7)) * u.strength, 0.0, 1.0);
    return vec4<f32>(mix(c.rgb, u.ink, edge), c.a);
  }
`;

/**
 * Builds the comic effect bound to one renderer. Call `enable(model)` to apply
 * the flat-material swap + outline pass to a loaded model, and `disable()` to
 * restore the original materials and remove the pass.
 */
export class ComicEffect {
  private outlinePass: ShaderPass;
  private swapped = new Map<Mesh, StandardMaterial | StandardMaterial[]>();
  private flatCache = new WeakMap<StandardMaterial, ShaderMaterial>();
  private whiteTex: Texture | null = null;
  private active = false;

  constructor(private renderer: { passes: ShaderPass[]; postProcessing: boolean }) {
    this.outlinePass = new ShaderPass({
      name: 'comic-outline',
      effect: OUTLINE_EFFECT,
      uniforms: {
        thickness: 1.5,
        strength: 1.0,
        colorEdge: 0.12,
        ink: new Color(0.04, 0.03, 0.03),
      },
      enabled: false,
    });
  }

  /** The pass is created once and lives in renderer.passes; toggled via enabled. */
  attach(): void {
    if (!this.renderer.passes.includes(this.outlinePass)) {
      this.renderer.passes.push(this.outlinePass);
    }
  }

  private white(): Texture {
    if (!this.whiteTex) {
      const c = document.createElement('canvas');
      c.width = c.height = 1;
      const ctx = c.getContext('2d')!;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, 1, 1);
      this.whiteTex = new Texture(c as unknown as ImageBitmap, { colorSpace: 'srgb' });
      this.whiteTex.needsUpdate();
    }
    return this.whiteTex;
  }

  private flatFor(mat: StandardMaterial): ShaderMaterial {
    let flat = this.flatCache.get(mat);
    if (!flat) {
      flat = new ShaderMaterial({
        surface: FLAT_SURFACE,
        uniforms: { tint: mat.color.clone(), albedo: mat.map ?? this.white() },
        side: mat.side,
      });
      this.flatCache.set(mat, flat);
    }
    return flat;
  }

  enable(model: Object3D): void {
    this.disable();
    model.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      if (Array.isArray(o.material)) {
        if (!o.material.every((m) => m instanceof StandardMaterial)) return;
        this.swapped.set(o, o.material as StandardMaterial[]);
        o.material = (o.material as StandardMaterial[]).map((m) => this.flatFor(m));
      } else if (o.material instanceof StandardMaterial) {
        this.swapped.set(o, o.material);
        o.material = this.flatFor(o.material);
      }
    });
    this.outlinePass.enabled = true;
    this.renderer.postProcessing = true;
    this.active = true;
  }

  disable(): void {
    for (const [mesh, original] of this.swapped) mesh.material = original;
    this.swapped.clear();
    this.outlinePass.enabled = false;
    this.active = false;
  }

  get isActive(): boolean {
    return this.active;
  }
}
