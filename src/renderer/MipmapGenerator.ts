/**
 * Generates mipmaps for a 2D texture by repeatedly rendering each level from
 * the previous one with a fullscreen-triangle blit. Pipelines are cached per
 * texture format.
 */
export class MipmapGenerator {
  private pipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  private sampler: GPUSampler;
  private module: GPUShaderModule;

  constructor(private device: GPUDevice) {
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
    this.module = device.createShaderModule({
      code: /* wgsl */ `
        struct VSOut {
          @builtin(position) pos : vec4<f32>,
          @location(0) uv : vec2<f32>,
        };
        @vertex
        fn vs(@builtin(vertex_index) i : u32) -> VSOut {
          var p = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
          var out : VSOut;
          out.pos = vec4<f32>(p[i], 0.0, 1.0);
          out.uv = (p[i] * vec2<f32>(0.5, -0.5)) + vec2<f32>(0.5, 0.5);
          return out;
        }
        @group(0) @binding(0) var src : texture_2d<f32>;
        @group(0) @binding(1) var smp : sampler;
        @fragment
        fn fs(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
          return textureSample(src, smp, uv);
        }
      `,
    });
  }

  private getPipeline(format: GPUTextureFormat): GPURenderPipeline {
    let pipeline = this.pipelines.get(format);
    if (!pipeline) {
      pipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: this.module, entryPoint: 'vs' },
        fragment: { module: this.module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelines.set(format, pipeline);
    }
    return pipeline;
  }

  static mipLevelCount(width: number, height: number): number {
    return Math.floor(Math.log2(Math.max(width, height))) + 1;
  }

  generate(texture: GPUTexture, format: GPUTextureFormat): void {
    if (texture.mipLevelCount <= 1) return;
    const pipeline = this.getPipeline(format);
    const encoder = this.device.createCommandEncoder();

    for (let level = 1; level < texture.mipLevelCount; level++) {
      const srcView = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
      const dstView = texture.createView({ baseMipLevel: level, mipLevelCount: 1 });

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: this.sampler },
        ],
      });

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: dstView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
  }
}
