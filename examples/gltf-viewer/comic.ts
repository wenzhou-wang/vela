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

// Outline effect: silhouette (depth occupancy) + interior creases (relative
// depth and reconstructed view normals) + texture color edges (3x3 Sobel).
// The scene is HDR-linear here (passes run before tonemap), so color deltas are
// measured in approximate gamma space to match perceptual edges.
const OUTLINE_EFFECT = /* wgsl */ `
  fn isGeo(d : f32) -> f32 { return select(1.0, 0.0, d >= 0.9999); }
  fn perceptual(c : vec3<f32>) -> vec3<f32> { return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)); }

  fn effect(uv : vec2<f32>) -> vec4<f32> {
    let center = sceneColor(uv);
    let inner = pp.resolution.zw * max(u.thickness, 0.5);
    let outerS = inner * 1.6;
    let dC = sceneDepth(uv);
    let object = isGeo(dC);

    // Silhouette: depth occupancy change across the 8-neighborhood (wider radius).
    var outer = 0.0;
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>(-outerS.x, 0.0)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>( outerS.x, 0.0)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>(0.0, -outerS.y)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>(0.0,  outerS.y)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>(-outerS.x, -outerS.y)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>( outerS.x,  outerS.y)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>(-outerS.x,  outerS.y)))));
    outer = max(outer, abs(object - isGeo(sceneDepth(uv + vec2<f32>( outerS.x, -outerS.y)))));

    // Inner 8-neighbor taps for depth/normal/color edges.
    let uvL = uv + vec2<f32>(-inner.x, 0.0); let uvR = uv + vec2<f32>(inner.x, 0.0);
    let uvU = uv + vec2<f32>(0.0, -inner.y); let uvD = uv + vec2<f32>(0.0, inner.y);
    let uvUL = uv + vec2<f32>(-inner.x, -inner.y); let uvUR = uv + vec2<f32>(inner.x, -inner.y);
    let uvDL = uv + vec2<f32>(-inner.x,  inner.y); let uvDR = uv + vec2<f32>(inner.x,  inner.y);
    let dL = sceneDepth(uvL); let dR = sceneDepth(uvR);
    let dU = sceneDepth(uvU); let dD = sceneDepth(uvD);
    let pC = viewPosition(uv, dC);
    let pL = viewPosition(uvL, dL); let pR = viewPosition(uvR, dR);
    let pU = viewPosition(uvU, dU); let pD = viewPosition(uvD, dD);
    let pUL = viewPosition(uvUL, sceneDepth(uvUL)); let pUR = viewPosition(uvUR, sceneDepth(uvUR));
    let pDL = viewPosition(uvDL, sceneDepth(uvDL)); let pDR = viewPosition(uvDR, sceneDepth(uvDR));
    let neighborsGeo = min(min(isGeo(dL), isGeo(dR)), min(isGeo(dU), isGeo(dD)));

    // Relative-depth crease (folds facing the camera).
    let relDepth = max(max(abs(pC.z - pL.z), abs(pC.z - pR.z)), max(abs(pC.z - pU.z), abs(pC.z - pD.z))) / max(abs(pC.z), 1e-3);
    let depthEdge = smoothstep(0.015, 0.05, relDepth) * object * neighborsGeo;

    // Normal crease (silhouette-internal form changes).
    let nC = normalize(cross(pR - pL, pD - pU));
    let nL = normalize(cross(pC - pL, pDL - pUL));
    let nR = normalize(cross(pR - pC, pDR - pUR));
    let nU = normalize(cross(pUR - pUL, pC - pU));
    let nD = normalize(cross(pDR - pDL, pD - pC));
    let normalDelta = max(max(1.0 - abs(dot(nC, nL)), 1.0 - abs(dot(nC, nR))), max(1.0 - abs(dot(nC, nU)), 1.0 - abs(dot(nC, nD))));
    let normalEdge = smoothstep(0.12, 0.3, normalDelta) * object * neighborsGeo;

    // Texture color edges (3x3 Sobel in gamma space) — draws painted features.
    let cL = perceptual(sceneColor(uvL).rgb); let cR = perceptual(sceneColor(uvR).rgb);
    let cU = perceptual(sceneColor(uvU).rgb); let cD = perceptual(sceneColor(uvD).rgb);
    let cUL = perceptual(sceneColor(uvUL).rgb); let cUR = perceptual(sceneColor(uvUR).rgb);
    let cDL = perceptual(sceneColor(uvDL).rgb); let cDR = perceptual(sceneColor(uvDR).rgb);
    let sobelX = -cUL - 2.0 * cL - cDL + cUR + 2.0 * cR + cDR;
    let sobelY = -cUL - 2.0 * cU - cUR + cDL + 2.0 * cD + cDR;
    let colorDelta = (length(sobelX) + length(sobelY)) * 0.25;
    let colorEdge = smoothstep(u.colorEdge, u.colorEdge * 2.5, colorDelta) * object;

    let inkEdge = max(outer, max(depthEdge, max(normalEdge, colorEdge)));
    let edge = clamp(inkEdge * u.strength, 0.0, 1.0);
    return vec4<f32>(mix(center.rgb, u.ink, edge), center.a);
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
        colorEdge: 0.08,
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
