import { POST_SHADER } from './shaders/post.wgsl';
import { OIT_ACCUM_FORMAT, OIT_REVEAL_FORMAT } from './constants';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const LDR_FORMAT: GPUTextureFormat = 'rgba8unorm';

export interface PostOptions {
  fxaa: boolean;
  bloom: boolean;
  bloomThreshold: number;
  bloomIntensity: number;
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
      ],
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const dummy = device.createTexture({
      size: [1, 1], format: HDR_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.dummyView = dummy.createView();
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
    this.hdrResolve = attach(HDR_FORMAT, [w, h]);
    this.hdrView = this.hdrResolve.createView();
    this.ldrPing = attach(LDR_FORMAT, [w, h]);
    this.ldrView = this.ldrPing.createView();
    // Half-resolution bloom targets (cheaper, naturally softer).
    const bw = Math.max(1, w >> 1), bh = Math.max(1, h >> 1);
    this.bloomA = attach(HDR_FORMAT, [bw, bh]);
    this.bloomAView = this.bloomA.createView();
    this.bloomB = attach(HDR_FORMAT, [bw, bh]);
    this.bloomBView = this.bloomB.createView();
    // OIT accumulation/revealage (full-res, sample count 1).
    this.oitAccum?.destroy();
    this.oitReveal?.destroy();
    this.oitAccum = attach(OIT_ACCUM_FORMAT, [w, h]);
    this.oitAccumView = this.oitAccum.createView();
    this.oitReveal = attach(OIT_REVEAL_FORMAT, [w, h]);
    this.oitRevealView = this.oitReveal.createView();
  }

  /** The HDR scene target (so the OIT pass can depth-test/composite against it). */
  get hdrTargetView(): GPUTextureView { return this.hdrView; }

  /** Color attachments for the OIT transparent pass (accum cleared to 0, reveal to 1). */
  oitColorAttachments(): GPURenderPassColorAttachment[] {
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

  /** Run the post chain from the HDR target into the swap-chain `output` view. */
  run(encoder: GPUCommandEncoder, output: GPUTextureView, opts: PostOptions): void {
    // params = (1/width, 1/height, bloomThreshold, bloomIntensity)
    this.device.queue.writeBuffer(this.paramsBuffer, 0,
      new Float32Array([1 / this.width, 1 / this.height, opts.bloomThreshold, opts.bloomIntensity]));

    let bloomView = this.dummyView;
    if (opts.bloom) {
      this.pass(encoder, 'fs_threshold', HDR_FORMAT, this.hdrView, this.bloomAView);
      this.pass(encoder, 'fs_blurH', HDR_FORMAT, this.bloomAView, this.bloomBView);
      this.pass(encoder, 'fs_blurV', HDR_FORMAT, this.bloomBView, this.bloomAView);
      bloomView = this.bloomAView;
    }

    const tonemapEntry = opts.bloom ? 'fs_tonemapBloom' : 'fs_tonemap';
    if (opts.fxaa) {
      this.pass(encoder, tonemapEntry, LDR_FORMAT, this.hdrView, this.ldrView, bloomView);
      this.pass(encoder, 'fs_fxaa', this.swapFormat, this.ldrView, output);
    } else {
      this.pass(encoder, tonemapEntry, this.swapFormat, this.hdrView, output, bloomView);
    }
  }

  private pass(
    encoder: GPUCommandEncoder,
    entry: string,
    targetFormat: GPUTextureFormat,
    input: GPUTextureView,
    output: GPUTextureView,
    bloomInput: GPUTextureView = this.dummyView,
  ): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: input },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: bloomInput },
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
