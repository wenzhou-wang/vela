import { POST_SHADER } from './shaders/post.wgsl';

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const LDR_FORMAT: GPUTextureFormat = 'rgba8unorm';

/**
 * A minimal post-processing pipeline / render graph: the scene renders into a
 * linear HDR offscreen target, then a chain of fullscreen passes (ACES tonemap,
 * optional FXAA) resolves to the swap chain. This is the foundation the richer
 * effects (bloom, SSAO, TAA) build on.
 */
export class PostProcessing {
  private module: GPUShaderModule;
  private bindLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private paramsBuffer: GPUBuffer;
  private pipelines = new Map<string, GPURenderPipeline>();

  private width = 0;
  private height = 0;
  private hdrMSAA: GPUTexture | null = null;
  private hdrResolve!: GPUTexture;
  private hdrView!: GPUTextureView;
  private ldrPing!: GPUTexture;
  private ldrView!: GPUTextureView;

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
      ],
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** (Re)allocate size-dependent targets. `w`/`h` are device pixels. */
  ensureSize(w: number, h: number): void {
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.hdrMSAA?.destroy();
    this.hdrResolve?.destroy();
    this.ldrPing?.destroy();

    this.hdrMSAA = this.sampleCount > 1
      ? this.device.createTexture({
          size: [w, h], format: HDR_FORMAT, sampleCount: this.sampleCount,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
      : null;
    this.hdrResolve = this.device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdrResolve.createView();
    this.ldrPing = this.device.createTexture({
      size: [w, h], format: LDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.ldrView = this.ldrPing.createView();

    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Float32Array([1, 1 / w, 1 / h, 0]));
  }

  /** Color attachment for the scene pass: renders into the HDR target. */
  sceneColorAttachment(clearValue: GPUColor): GPURenderPassColorAttachment {
    return this.sampleCount > 1
      ? { view: this.hdrMSAA!.createView(), resolveTarget: this.hdrView, loadOp: 'clear', storeOp: 'store', clearValue }
      : { view: this.hdrView, loadOp: 'clear', storeOp: 'store', clearValue };
  }

  /** Run the post chain from the HDR target into the swap-chain `output` view. */
  run(encoder: GPUCommandEncoder, output: GPUTextureView, fxaa: boolean): void {
    if (fxaa) {
      this.pass(encoder, 'fs_tonemap', LDR_FORMAT, this.hdrView, this.ldrView);
      this.pass(encoder, 'fs_fxaa', this.swapFormat, this.ldrView, output);
    } else {
      this.pass(encoder, 'fs_tonemap', this.swapFormat, this.hdrView, output);
    }
  }

  private pass(
    encoder: GPUCommandEncoder,
    entry: string,
    targetFormat: GPUTextureFormat,
    input: GPUTextureView,
    output: GPUTextureView,
  ): void {
    const pipeline = this.pipeline(entry, targetFormat);
    const bindGroup = this.device.createBindGroup({
      layout: this.bindLayout,
      entries: [
        { binding: 0, resource: input },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: output, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(pipeline);
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
