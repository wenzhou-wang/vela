import { POST_SHADER } from './shaders/post.wgsl';
import { SSAO_SHADER } from './shaders/ssao.wgsl';
import { OIT_ACCUM_FORMAT, OIT_REVEAL_FORMAT } from './constants';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const LDR_FORMAT: GPUTextureFormat = 'rgba8unorm';

export interface PostOptions {
  fxaa: boolean;
  bloom: boolean;
  bloomThreshold: number;
  bloomIntensity: number;
  ssao: boolean;
  ssaoStrength: number;
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

  // SSAO resources
  private ssaoModule: GPUShaderModule;
  private ssaoBindLayout: GPUBindGroupLayout;
  private ssaoParamsBuffer: GPUBuffer;
  private ssaoA: GPUTexture | null = null;
  private ssaoAView!: GPUTextureView;
  private ssaoB: GPUTexture | null = null;
  private ssaoBView!: GPUTextureView;

  private width = 0;
  private height = 0;
  private hdrMSAA: GPUTexture | null = null;
  private hdrResolve!: GPUTexture;
  private hdrView!: GPUTextureView;
  private ldrPing!: GPUTexture;
  private ldrView!: GPUTextureView;
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
    // 32 bytes: (1/w, 1/h, bloomThr, bloomInt) + (ssaoStrength, 0, 0, 0)
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
  }

  /** The HDR scene target (so the OIT pass can depth-test/composite against it). */
  get hdrTargetView(): GPUTextureView { return this.hdrView; }

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

  /** Run the post chain from the HDR target into the swap-chain `output` view. */
  run(encoder: GPUCommandEncoder, output: GPUTextureView, opts: PostOptions): void {
    // params0 = (1/width, 1/height, bloomThreshold, bloomIntensity)
    // params1 = (ssaoStrength, 0, 0, 0)
    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Float32Array([
      1 / this.width, 1 / this.height, opts.bloomThreshold, opts.bloomIntensity,
      opts.ssao ? opts.ssaoStrength : 0, 0, 0, 0,
    ]));

    let bloomView = this.dummyView;
    if (opts.bloom) {
      this.pass(encoder, 'fs_threshold', HDR_FORMAT, this.hdrView, this.bloomAView);
      this.pass(encoder, 'fs_blurH', HDR_FORMAT, this.bloomAView, this.bloomBView);
      this.pass(encoder, 'fs_blurV', HDR_FORMAT, this.bloomBView, this.bloomAView);
      bloomView = this.bloomAView;
    }

    const ssaoView = opts.ssao ? this.ssaoAView : this.dummyWhiteView;
    const tonemapEntry = opts.bloom ? 'fs_tonemapBloom' : 'fs_tonemap';
    if (opts.fxaa) {
      this.pass(encoder, tonemapEntry, LDR_FORMAT, this.hdrView, this.ldrView, bloomView, ssaoView);
      this.pass(encoder, 'fs_fxaa', this.swapFormat, this.ldrView, output);
    } else {
      this.pass(encoder, tonemapEntry, this.swapFormat, this.hdrView, output, bloomView, ssaoView);
    }
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
