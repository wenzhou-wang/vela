import { POST_SHADER } from './shaders/post.wgsl';
import { SSAO_SHADER } from './shaders/ssao.wgsl';
import { TAA_SHADER } from './shaders/taa.wgsl';
import { LUT_SHADER } from './shaders/lut.wgsl';
import { buildShaderPass } from './shaders/shaderpass.wgsl';
import { computeUniformLayout, packUniforms, type UniformLayout } from '../materials/ShaderMaterial';
import { floatToHalfArray } from './TextureManager';
import type { ShaderPass } from './ShaderPass';
import type { Texture } from '../textures/Texture';
import type { ColorLUT } from '../textures/ColorLUT';
import type { TextureManager } from './TextureManager';
import { DEPTH_FORMAT, OIT_ACCUM_FORMAT, OIT_REVEAL_FORMAT } from './constants';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const LDR_FORMAT: GPUTextureFormat = 'rgba8unorm';

/** Per-ShaderPass cached GPU resources (pipeline + group-1 uniform binding). */
interface ShaderPassResources {
  pipeline: GPURenderPipeline;
  version: number;             // material version the pipeline was built for
  shapeKey: string;            // uniform shape the pipeline + layout assume
  buffer: GPUBuffer | null;
  data: Float32Array<ArrayBuffer>;
  layout: UniformLayout;
  bindGroup1: GPUBindGroup | null;  // null when the pass has no group-1 uniforms
  textureSig: string;
}

/** Output transform applied at the end of the post chain. */
export type ToneMapping = 'aces' | 'agx' | 'none';

export interface PostOptions {
  fxaa: boolean;
  bloom: boolean;
  bloomThreshold: number;
  bloomIntensity: number;
  ssao: boolean;
  ssaoStrength: number;
  /** 'aces' = filmic curve (default); 'agx' = AgX operator; 'none' = sRGB-only (flat/stylized). */
  toneMapping: ToneMapping;
  /** Optional 3-D color-grading LUT applied after tonemap (display space). */
  colorLUT?: ColorLUT | null;
}

/**
 * A minimal post-processing pipeline / render graph: the scene renders into a
 * linear HDR offscreen target, then a chain of fullscreen passes (bright-pass +
 * separable-Gaussian bloom, ACES tonemap, optional FXAA) resolves to the swap
 * chain. The foundation the richer effects build on.
 */
export class PostProcessing {
  private module: GPUShaderModule;
  private bindLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private paramsBuffer: GPUBuffer;
  private pipelines = new Map<string, GPURenderPipeline>();
  private dummyView: GPUTextureView;
  private dummyWhiteView: GPUTextureView;
  private dummyDepthView: GPUTextureView;

  // SSAO resources
  private ssaoModule: GPUShaderModule;
  private ssaoBindLayout: GPUBindGroupLayout;
  private ssaoParamsBuffer: GPUBuffer;
  private ssaoA: GPUTexture | null = null;
  private ssaoAView!: GPUTextureView;
  private ssaoB: GPUTexture | null = null;
  private ssaoBView!: GPUTextureView;

  // TAA resources: ping-pong history/output targets + params.
  private taaModule: GPUShaderModule;
  private taaBindLayout: GPUBindGroupLayout;
  private taaParamsBuffer: GPUBuffer;
  private taaA: GPUTexture | null = null;
  private taaAView!: GPUTextureView;
  private taaB: GPUTexture | null = null;
  private taaBView!: GPUTextureView;
  private taaFlip = false;
  private taaHistoryValid = false;

  private width = 0;
  private height = 0;
  private hdrMSAA: GPUTexture | null = null;
  private hdrResolve!: GPUTexture;
  private hdrView!: GPUTextureView;
  private ldrPing!: GPUTexture;
  private ldrView!: GPUTextureView;
  private ldrPong!: GPUTexture;
  private ldrView2!: GPUTextureView;
  private bloomA!: GPUTexture;
  private bloomAView!: GPUTextureView;
  private bloomB!: GPUTexture;
  private bloomBView!: GPUTextureView;
  private oitAccum: GPUTexture | null = null;
  private oitAccumView!: GPUTextureView;
  private oitReveal: GPUTexture | null = null;
  private oitRevealView!: GPUTextureView;
  // MSAA OIT targets (when sampleCount > 1): resolve into the non-MSAA views above.
  private oitAccumMSAA: GPUTexture | null = null;
  private oitRevealMSAA: GPUTexture | null = null;
  // Screen-space refraction: a copy of the HDR target captured before transparent draws.
  private sceneCapture: GPUTexture | null = null;
  private _sceneCaptureView!: GPUTextureView;

  // Custom ShaderPass ping-pong (full-res HDR) + per-pass GPU resource cache.
  private passA: GPUTexture | null = null;
  private passAView!: GPUTextureView;
  private passB: GPUTexture | null = null;
  private passBView!: GPUTextureView;
  private passBindLayout: GPUBindGroupLayout;
  private passParamsBuffer: GPUBuffer;

  // Color-grading LUT resources (lazily built 3-D texture).
  private lutModule: GPUShaderModule;
  private lutBindLayout: GPUBindGroupLayout;
  private lutParamsBuffer: GPUBuffer;
  private lutTexture: GPUTexture | null = null;
  private lutView: GPUTextureView | null = null;
  private lutSource: ColorLUT | null = null;
  private lutSize = 0;
  private lutVersion = -1;
  private passResources = new Map<string, ShaderPassResources>();

  constructor(
    private device: GPUDevice,
    private swapFormat: GPUTextureFormat,
    private sampleCount: number,
  ) {
    this.module = device.createShaderModule({ code: POST_SHADER, label: 'post' });
    this.bindLayout = device.createBindGroupLayout({
      label: 'post',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    // 32 bytes: data + ssao parameter blocks.
    this.paramsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const dummy = device.createTexture({
      size: [1, 1], format: HDR_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.dummyView = dummy.createView();

    // 1×1 white texture used as the SSAO dummy when SSAO is disabled.
    const whiteTex = device.createTexture({
      size: [1, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: whiteTex }, new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 }, [1, 1]);
    this.dummyWhiteView = whiteTex.createView();

    const dummyDepth = device.createTexture({
      size: [1, 1], format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.dummyDepthView = dummyDepth.createView({ aspect: 'depth-only' });
    const depthEncoder = device.createCommandEncoder();
    const depthPass = depthEncoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.dummyDepthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    depthPass.end();
    device.queue.submit([depthEncoder.finish()]);

    // Dummy scene-capture (1×1) — replaced by ensureSize(); keeps sceneCaptureView valid.
    const capDummy = device.createTexture({
      size: [1, 1], format: HDR_FORMAT,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._sceneCaptureView = capDummy.createView();

    // SSAO resources
    this.ssaoModule = device.createShaderModule({ code: SSAO_SHADER, label: 'ssao' });
    this.ssaoBindLayout = device.createBindGroupLayout({
      label: 'ssao',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    // invProj(64) + proj(64) + params(16) = 144 bytes
    this.ssaoParamsBuffer = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // TAA resources
    this.taaModule = device.createShaderModule({ code: TAA_SHADER, label: 'taa' });
    this.taaBindLayout = device.createBindGroupLayout({
      label: 'taa',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    // prevViewProj(64) + invViewProj(64) + info(16) = 144 bytes
    this.taaParamsBuffer = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Custom ShaderPass group-0 layout: scene color + sampler + depth + params.
    this.passBindLayout = device.createBindGroupLayout({
      label: 'shaderpass',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    // resolution(16) + time(16) = 32 bytes
    // resolution + time (32) + proj (64) + invProj (64) + camera near/far (16) = 176.
    this.passParamsBuffer = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Color-grading LUT pass (3-D texture needs its own bind layout).
    this.lutModule = device.createShaderModule({ code: LUT_SHADER, label: 'lut' });
    this.lutBindLayout = device.createBindGroupLayout({
      label: 'lut',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    // info(16) + domain min/max (vec4 each, 32) = 48 bytes.
    this.lutParamsBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** (Re)allocate size-dependent targets. `w`/`h` are device pixels. */
  ensureSize(w: number, h: number): void {
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.hdrMSAA?.destroy();
    this.hdrResolve?.destroy();
    this.ldrPing?.destroy();
    this.bloomA?.destroy();
    this.bloomB?.destroy();

    const attach = (format: GPUTextureFormat, size: [number, number], sampleCount = 1): GPUTexture =>
      this.device.createTexture({
        size, format, sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT |
          (sampleCount > 1 ? 0 : GPUTextureUsage.TEXTURE_BINDING),
      });

    this.hdrMSAA = this.sampleCount > 1 ? attach(HDR_FORMAT, [w, h], this.sampleCount) : null;
    // hdrResolve needs COPY_SRC so we can snapshot it for screen-space refraction.
    this.hdrResolve = this.device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.hdrView = this.hdrResolve.createView();
    // Screen-space refraction capture (same size, COPY_DST | TEXTURE_BINDING).
    this.sceneCapture?.destroy();
    this.sceneCapture = this.device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    this._sceneCaptureView = this.sceneCapture.createView();
    this.ldrPing = attach(LDR_FORMAT, [w, h]);
    this.ldrView = this.ldrPing.createView();
    this.ldrPong?.destroy();
    this.ldrPong = attach(LDR_FORMAT, [w, h]);
    this.ldrView2 = this.ldrPong.createView();
    // Half-resolution bloom targets (cheaper, naturally softer).
    const bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
    this.bloomA = attach(HDR_FORMAT, [bw, bh]);
    this.bloomAView = this.bloomA.createView();
    this.bloomB = attach(HDR_FORMAT, [bw, bh]);
    this.bloomBView = this.bloomB.createView();
    // OIT accumulation/revealage (full-res).  When MSAA is on, create MSAA render
    // targets that resolve into the non-MSAA TEXTURE_BINDING views used by compositeOIT.
    this.oitAccum?.destroy();
    this.oitReveal?.destroy();
    this.oitAccumMSAA?.destroy();
    this.oitRevealMSAA?.destroy();
    this.oitAccum = attach(OIT_ACCUM_FORMAT, [w, h]);
    this.oitAccumView = this.oitAccum.createView();
    this.oitReveal = attach(OIT_REVEAL_FORMAT, [w, h]);
    this.oitRevealView = this.oitReveal.createView();
    this.oitAccumMSAA  = this.sampleCount > 1 ? attach(OIT_ACCUM_FORMAT,  [w, h], this.sampleCount) : null;
    this.oitRevealMSAA = this.sampleCount > 1 ? attach(OIT_REVEAL_FORMAT, [w, h], this.sampleCount) : null;

    // SSAO ping-pong targets (full-res HDR, using r channel for occlusion).
    this.ssaoA?.destroy();
    this.ssaoB?.destroy();
    this.ssaoA = attach(HDR_FORMAT, [w, h]);
    this.ssaoAView = this.ssaoA.createView();
    this.ssaoB = attach(HDR_FORMAT, [w, h]);
    this.ssaoBView = this.ssaoB.createView();

    // TAA history/output ping-pong (full-res HDR). Resizing invalidates history.
    this.taaA?.destroy();
    this.taaB?.destroy();
    this.taaA = attach(HDR_FORMAT, [w, h]);
    this.taaAView = this.taaA.createView();
    this.taaB = attach(HDR_FORMAT, [w, h]);
    this.taaBView = this.taaB.createView();
    this.taaHistoryValid = false;

    // Custom ShaderPass ping-pong (full-res HDR; lazily used).
    this.passA?.destroy();
    this.passB?.destroy();
    this.passA = attach(HDR_FORMAT, [w, h]);
    this.passAView = this.passA.createView();
    this.passB = attach(HDR_FORMAT, [w, h]);
    this.passBView = this.passB.createView();
  }

  /** The HDR scene target (so the OIT pass can depth-test/composite against it). */
  get hdrTargetView(): GPUTextureView { return this.hdrView; }

  /** A 1×1 cleared depth view, for passes that have no real scene depth. */
  get dummyDepth(): GPUTextureView { return this.dummyDepthView; }

  /** View of the HDR snapshot taken before transparent draws (screen-space refraction). */
  get sceneCaptureView(): GPUTextureView { return this._sceneCaptureView; }

  /** Copy the current HDR render target into the scene-capture texture. */
  captureHDR(encoder: GPUCommandEncoder): void {
    if (!this.sceneCapture) return;
    const [w, h] = [this.width, this.height];
    encoder.copyTextureToTexture(
      { texture: this.hdrResolve },
      { texture: this.sceneCapture },
      [w, h],
    );
  }

  /** Color attachments for the OIT transparent pass (accum cleared to 0, reveal to 1). */
  oitColorAttachments(): GPURenderPassColorAttachment[] {
    if (this.sampleCount > 1 && this.oitAccumMSAA && this.oitRevealMSAA) {
      // MSAA path: render into MSAA targets, resolve immediately into the non-MSAA
      // views that compositeOIT() will sample from.
      return [
        {
          view: this.oitAccumMSAA.createView(),
          resolveTarget: this.oitAccumView,
          loadOp: 'clear', storeOp: 'discard',
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        },
        {
          view: this.oitRevealMSAA.createView(),
          resolveTarget: this.oitRevealView,
          loadOp: 'clear', storeOp: 'discard',
          clearValue: { r: 1, g: 1, b: 1, a: 1 },
        },
      ];
    }
    return [
      { view: this.oitAccumView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } },
      { view: this.oitRevealView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 1, b: 1, a: 1 } },
    ];
  }

  /** Composite the OIT accum/reveal targets onto the HDR scene target. */
  compositeOIT(encoder: GPUCommandEncoder): void {
    const pipeline = this.blendPipeline('fs_oitComposite', HDR_FORMAT);
    const bindGroup = this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: this.oitAccumView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: this.oitRevealView },
        { binding: 4, resource: this.dummyWhiteView },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.hdrView, loadOp: 'load', storeOp: 'store' }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  private blendPipeline(entry: string, targetFormat: GPUTextureFormat): GPURenderPipeline {
    const key = `${entry}|${targetFormat}|blend`;
    let p = this.pipelines.get(key);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_main' },
      fragment: {
        module: this.module,
        entryPoint: entry,
        targets: [{
          format: targetFormat,
          blend: {
            color: { srcFactor: 'one-minus-src-alpha', dstFactor: 'src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one-minus-src-alpha', dstFactor: 'src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.pipelines.set(key, p);
    return p;
  }

  /** Color attachment for the scene pass: renders into the HDR target. */
  sceneColorAttachment(clearValue: GPUColor): GPURenderPassColorAttachment {
    return this.sampleCount > 1
      ? { view: this.hdrMSAA!.createView(), resolveTarget: this.hdrView, loadOp: 'clear', storeOp: 'store', clearValue }
      : { view: this.hdrView, loadOp: 'clear', storeOp: 'store', clearValue };
  }

  /**
   * Render the SSAO occlusion pass then blur it.  Call this AFTER the scene
   * render pass (depth must be populated) and BEFORE `run()`.
   * `depthView` must be a depth-only view of a non-MSAA depth32float texture.
   * `invProj` and `proj` are Float32Arrays of 16 elements each (column-major).
   */
  runSSAO(
    encoder: GPUCommandEncoder,
    depthView: GPUTextureView,
    invProj: Float32Array,
    proj: Float32Array,
    radius: number,
    bias: number,
  ): void {
    // invProj(64 bytes) + proj(64 bytes) + params(16 bytes)
    const data = new Float32Array(36);
    data.set(invProj, 0);
    data.set(proj, 16);
    data[32] = radius;
    data[33] = bias;
    data[34] = 1 / this.width;
    data[35] = 1 / this.height;
    this.device.queue.writeBuffer(this.ssaoParamsBuffer, 0, data);

    const bg = this.device.createBindGroup({
      layout: this.ssaoBindLayout,
      entries: [
        { binding: 0, resource: depthView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.ssaoParamsBuffer } },
      ],
    });

    const ssaoPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.ssaoAView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 1, g: 1, b: 1, a: 1 },
      }],
    });
    ssaoPass.setPipeline(this.ssaoPipeline());
    ssaoPass.setBindGroup(0, bg);
    ssaoPass.draw(3);
    ssaoPass.end();

    // One-pass bilateral-ish blur (Gaussian, re-uses existing blur entries).
    this.pass(encoder, 'fs_blurH', HDR_FORMAT, this.ssaoAView, this.ssaoBView);
    this.pass(encoder, 'fs_blurV', HDR_FORMAT, this.ssaoBView, this.ssaoAView);
  }

  private ssaoPipeline(): GPURenderPipeline {
    const key = 'ssao';
    let p = this.pipelines.get(key);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.ssaoBindLayout] }),
      vertex:   { module: this.ssaoModule, entryPoint: 'vs_main' },
      fragment: { module: this.ssaoModule, entryPoint: 'fs_ssao', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipelines.set(key, p);
    return p;
  }

  /**
   * Temporal AA resolve: blend the (jittered) HDR scene with the reprojected
   * history, writing the result into a ping-pong target whose view is returned —
   * pass it to `run()` as the post-chain input.  Call AFTER the scene/OIT passes
   * and BEFORE `run()`.  `depthView` must be a non-MSAA depth32float view.
   * `prevViewProj`/`invViewProj` are 16-element column-major Float32Arrays;
   * `jitterX`/`jitterY` are the NDC jitter baked into this frame's projection.
   */
  runTAA(
    encoder: GPUCommandEncoder,
    depthView: GPUTextureView,
    prevViewProj: Float32Array,
    invViewProj: Float32Array,
    jitterX: number,
    jitterY: number,
    blend: number,
  ): GPUTextureView {
    const data = new Float32Array(36);
    data.set(prevViewProj, 0);
    data.set(invViewProj, 16);
    data[32] = jitterX;
    data[33] = jitterY;
    data[34] = this.taaHistoryValid ? blend : 1.0; // first frame: take current as-is
    this.device.queue.writeBuffer(this.taaParamsBuffer, 0, data);

    const historyView = this.taaFlip ? this.taaBView : this.taaAView;
    const outputView = this.taaFlip ? this.taaAView : this.taaBView;
    const bg = this.device.createBindGroup({
      layout: this.taaBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.taaParamsBuffer } },
        { binding: 1, resource: this.hdrView },
        { binding: 2, resource: historyView },
        { binding: 3, resource: depthView },
        { binding: 4, resource: this.sampler },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: outputView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.taaPipeline());
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();

    this.taaFlip = !this.taaFlip;
    this.taaHistoryValid = true;
    return outputView;
  }

  /** Drop the TAA history (e.g. when TAA is toggled off) so re-enabling starts clean. */
  invalidateTAAHistory(): void {
    this.taaHistoryValid = false;
  }

  /**
   * Run the enabled custom ShaderPasses in order over `input` (HDR linear),
   * ping-ponging between the two pass buffers, and return the final HDR view to
   * feed the tonemap chain. `depthView` is the scene depth (or a dummy).
   * `textures` resolves a pass's texture uniforms; `time` drives `pp.time`.
   * Returns `input` unchanged when no pass is enabled.
   */
  runShaderPasses(
    encoder: GPUCommandEncoder,
    passes: ShaderPass[],
    input: GPUTextureView,
    depthView: GPUTextureView,
    textures: TextureManager,
    time: number,
    camera: { proj: Float32Array; invProj: Float32Array; near: number; far: number },
  ): GPUTextureView {
    const active = passes.filter((p) => p.enabled);
    if (active.length === 0) return input;

    // PassParams: resolution(4) + time(4) + proj(16) + invProj(16) + camera(4) = 44 floats.
    const params = new Float32Array(44);
    params.set([this.width, this.height, 1 / this.width, 1 / this.height, time, 0, 0, 0], 0);
    params.set(camera.proj, 8);
    params.set(camera.invProj, 24);
    params[40] = camera.near;
    params[41] = camera.far;
    this.device.queue.writeBuffer(this.passParamsBuffer, 0, params);

    let src = input;
    const buffers = [this.passAView, this.passBView];
    let writeIdx = 0;
    for (const pass of active) {
      const res = this.getShaderPassResources(pass, textures);
      // Avoid reading and writing the same texture: if src is the buffer we'd
      // write, flip to the other one.
      let dst = buffers[writeIdx];
      if (dst === src) { writeIdx ^= 1; dst = buffers[writeIdx]; }

      const group0 = this.device.createBindGroup({
        layout: this.passBindLayout,
        entries: [
          { binding: 0, resource: src },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: depthView },
          { binding: 3, resource: { buffer: this.passParamsBuffer } },
        ],
      });
      const rp = encoder.beginRenderPass({
        colorAttachments: [{ view: dst, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      });
      rp.setPipeline(res.pipeline);
      rp.setBindGroup(0, group0);
      if (res.bindGroup1) rp.setBindGroup(1, res.bindGroup1);
      rp.draw(3);
      rp.end();

      src = dst;
      writeIdx ^= 1;
    }
    return src;
  }

  /** Build/refresh a ShaderPass's pipeline + group-1 uniform binding. */
  private getShaderPassResources(pass: ShaderPass, textures: TextureManager): ShaderPassResources {
    const layout = computeUniformLayout(pass.uniforms, 1);
    const shapeKey = layout.fields.map((f) => `${f.name}:${f.kind}`).join(',') + '|tex:' + layout.textures.join(',');
    let textureSig = '';
    for (const name of layout.textures) {
      const tex = pass.uniforms[name] as Texture;
      textureSig += `${name}=${tex.id}:${tex.version};`;
    }

    const existing = this.passResources.get(pass.id);
    const needsRebuild = !existing || existing.version !== pass.version || existing.shapeKey !== shapeKey;
    let res: ShaderPassResources;
    if (!existing || needsRebuild) {
      existing?.buffer?.destroy();
      const code = buildShaderPass(layout.wgsl, pass.effectCode);
      const module = this.device.createShaderModule({ code, label: `shaderpass:${pass.name}` });
      module.getCompilationInfo().then((info) => {
        const errs = info.messages.filter((m) => m.type === 'error');
        if (errs.length === 0) return;
        const lines = code.split('\n');
        let report = `[vela] ShaderPass "${pass.name}" failed to compile:\n`;
        for (const m of errs) {
          report += `  :${m.lineNum}:${m.linePos} ${m.message}\n`;
          if (lines[m.lineNum - 1] !== undefined) report += `    ${m.lineNum} | ${lines[m.lineNum - 1]}\n`;
        }
        console.error(report + 'Fix the `effect` WGSL (fn effect(uv : vec2<f32>) -> vec4<f32>).');
      });
      const layouts = [this.passBindLayout];
      if (layout.fields.length > 0 || layout.textures.length > 0) {
        layouts.push(this.customGroup1Layout(layout.fields.length > 0, layout.textures.length));
      }
      const pipeline = this.device.createRenderPipeline({
        label: `shaderpass:${pass.name}`,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: layouts }),
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
        primitive: { topology: 'triangle-list' },
      });
      const buffer = layout.fields.length > 0
        ? this.device.createBuffer({ size: layout.size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
        : null;
      res = {
        pipeline, version: pass.version, shapeKey, buffer,
        data: new Float32Array(layout.size / 4), layout, bindGroup1: null, textureSig: '',
      };
      this.passResources.set(pass.id, res);
    } else {
      res = existing;
    }

    // (Re)build the group-1 bind group when shape/textures change.
    if (res.bindGroup1 === null || res.textureSig !== textureSig || needsRebuild) {
      if (layout.fields.length > 0 || layout.textures.length > 0) {
        const entries: GPUBindGroupEntry[] = [];
        if (res.buffer) entries.push({ binding: 0, resource: { buffer: res.buffer } });
        layout.textures.forEach((name, i) => {
          const entry = textures.get(pass.uniforms[name] as Texture);
          entries.push({ binding: 1 + i * 2, resource: entry.view });
          entries.push({ binding: 2 + i * 2, resource: entry.sampler });
        });
        res.bindGroup1 = this.device.createBindGroup({
          layout: this.customGroup1Layout(layout.fields.length > 0, layout.textures.length),
          entries,
        });
      }
      res.textureSig = textureSig;
    }
    if (res.buffer) {
      packUniforms(pass.uniforms, layout, res.data);
      this.device.queue.writeBuffer(res.buffer, 0, res.data);
    }
    return res;
  }

  private group1Layouts = new Map<number, GPUBindGroupLayout>();
  /** ShaderPass group-1 bind layout (uniforms + textures), cached by shape. */
  private customGroup1Layout(hasBuffer: boolean, textureCount: number): GPUBindGroupLayout {
    const shape = (hasBuffer ? 1000 : 0) + textureCount;
    let l = this.group1Layouts.get(shape);
    if (l) return l;
    const entries: GPUBindGroupLayoutEntry[] = [];
    if (hasBuffer) entries.push({ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
    for (let i = 0; i < textureCount; i++) {
      entries.push({ binding: 1 + i * 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } });
      entries.push({ binding: 2 + i * 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } });
    }
    l = this.device.createBindGroupLayout({ label: `shaderpass-g1-${shape}`, entries });
    this.group1Layouts.set(shape, l);
    return l;
  }

  private taaPipeline(): GPURenderPipeline {
    const key = 'taa';
    let p = this.pipelines.get(key);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.taaBindLayout] }),
      vertex:   { module: this.taaModule, entryPoint: 'vs_main' },
      fragment: { module: this.taaModule, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipelines.set(key, p);
    return p;
  }

  /**
   * Run the post chain from the HDR target (or `input`, e.g. the TAA output)
   * into the swap-chain `output` view.
   */
  run(
    encoder: GPUCommandEncoder,
    output: GPUTextureView,
    opts: PostOptions,
    input: GPUTextureView = this.hdrView,
  ): void {
    // data = (1/width, 1/height, bloomThreshold, bloomIntensity); ssao.x = strength.
    const params = new Float32Array(8);
    params.set([
      1 / this.width, 1 / this.height, opts.bloomThreshold, opts.bloomIntensity,
      opts.ssao ? opts.ssaoStrength : 0, 0, 0, 0,
    ]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    let bloomView = this.dummyView;
    if (opts.bloom) {
      this.pass(encoder, 'fs_threshold', HDR_FORMAT, input, this.bloomAView);
      this.pass(encoder, 'fs_blurH', HDR_FORMAT, this.bloomAView, this.bloomBView);
      this.pass(encoder, 'fs_blurV', HDR_FORMAT, this.bloomBView, this.bloomAView);
      bloomView = this.bloomAView;
    }

    const ssaoView = opts.ssao ? this.ssaoAView : this.dummyWhiteView;
    // 'none' skips the curve entirely; 'agx'/'aces' pick their bloom-paired or
    // plain tonemap entry (bloom is added in HDR before the curve).
    let baseEntry: string;
    if (opts.toneMapping === 'none') baseEntry = 'fs_linear';
    else if (opts.toneMapping === 'agx') baseEntry = opts.bloom ? 'fs_tonemapAgxBloom' : 'fs_tonemapAgx';
    else baseEntry = opts.bloom ? 'fs_tonemapBloom' : 'fs_tonemap';
    const lut = opts.colorLUT ?? null;
    if (lut) {
      // Tonemap (+ optional FXAA) into an LDR target, then grade it into the swap chain.
      this.pass(encoder, baseEntry, LDR_FORMAT, input, this.ldrView, bloomView, ssaoView);
      if (opts.fxaa) {
        this.pass(encoder, 'fs_fxaa', LDR_FORMAT, this.ldrView, this.ldrView2);
        this.runLUT(encoder, this.ldrView2, output, lut);
      } else {
        this.runLUT(encoder, this.ldrView, output, lut);
      }
    } else if (opts.fxaa) {
      this.pass(encoder, baseEntry, LDR_FORMAT, input, this.ldrView, bloomView, ssaoView);
      this.pass(encoder, 'fs_fxaa', this.swapFormat, this.ldrView, output);
    } else {
      this.pass(encoder, baseEntry, this.swapFormat, input, output, bloomView, ssaoView);
    }
  }

  /** Build/refresh the 3-D LUT texture from `lut` (keyed by identity + version). */
  private ensureLUT(lut: ColorLUT): void {
    if (this.lutSource === lut && this.lutVersion === lut.version && this.lutTexture) return;
    const n = lut.size;
    if (this.lutTexture && this.lutSize !== n) { this.lutTexture.destroy(); this.lutTexture = null; }
    if (!this.lutTexture) {
      this.lutTexture = this.device.createTexture({
        size: [n, n, n], dimension: '3d', format: HDR_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.lutView = this.lutTexture.createView({ dimension: '3d' });
      this.lutSize = n;
    }
    // Expand RGB → RGBA and upload as rgba16float (x = R fastest, matching .cube order).
    const rgba = new Float32Array(n * n * n * 4);
    for (let i = 0, j = 0; i < lut.data.length; i += 3, j += 4) {
      rgba[j] = lut.data[i]; rgba[j + 1] = lut.data[i + 1]; rgba[j + 2] = lut.data[i + 2]; rgba[j + 3] = 1;
    }
    this.device.queue.writeTexture(
      { texture: this.lutTexture },
      floatToHalfArray(rgba),
      { bytesPerRow: n * 8, rowsPerImage: n },
      [n, n, n],
    );
    this.lutSource = lut;
    this.lutVersion = lut.version;
  }

  /** Final color-grading pass: sample `input` (tonemapped LDR), grade through the LUT. */
  private runLUT(encoder: GPUCommandEncoder, input: GPUTextureView, output: GPUTextureView, lut: ColorLUT): void {
    this.ensureLUT(lut);
    const dmin = lut.domainMin, dmax = lut.domainMax;
    this.device.queue.writeBuffer(this.lutParamsBuffer, 0, new Float32Array([
      lut.strength, lut.size, 0, 0,
      dmin[0], dmin[1], dmin[2], 0,
      dmax[0], dmax[1], dmax[2], 0,
    ]));
    const bg = this.device.createBindGroup({
      layout: this.lutBindLayout,
      entries: [
        { binding: 0, resource: input },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.lutView! },
        { binding: 3, resource: { buffer: this.lutParamsBuffer } },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.lutPipeline(this.swapFormat));
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  private lutPipeline(format: GPUTextureFormat): GPURenderPipeline {
    const key = `lut|${format}`;
    let p = this.pipelines.get(key);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.lutBindLayout] }),
      vertex: { module: this.lutModule, entryPoint: 'vs_main' },
      fragment: { module: this.lutModule, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipelines.set(key, p);
    return p;
  }

  private pass(
    encoder: GPUCommandEncoder,
    entry: string,
    targetFormat: GPUTextureFormat,
    input: GPUTextureView,
    output: GPUTextureView,
    bloomInput: GPUTextureView = this.dummyView,
    ssaoInput: GPUTextureView = this.dummyWhiteView,
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: input },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: bloomInput },
        { binding: 4, resource: ssaoInput },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline(entry, targetFormat));
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  private pipeline(entry: string, targetFormat: GPUTextureFormat): GPURenderPipeline {
    const key = `${entry}|${targetFormat}`;
    let p = this.pipelines.get(key);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] }),
      vertex: { module: this.module, entryPoint: 'vs_main' },
      fragment: { module: this.module, entryPoint: entry, targets: [{ format: targetFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipelines.set(key, p);
    return p;
  }
}
