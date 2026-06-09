import { IBL_SHADER } from './shaders/ibl.wgsl';

/** Number of roughness mip levels in the prefiltered specular map. */
export const IBL_MIP_LEVELS = 6;
/** Width of the prefiltered specular base level. */
export const IBL_SPEC_W = 256;
/** Height of the prefiltered specular base level. */
export const IBL_SPEC_H = 128;

/**
 * Manages GPU-side IBL prefilter resources:
 *   - `brdfLUT`        128×128 RGBA16F  (NoV, roughness) → (F_scale, F_bias)
 *   - `irradiance`     64×32 RGBA16F    cosine-weighted env convolution
 *   - `prefiltered`    256×128 RGBA16F  GGX-prefiltered specular (IBL_MIP_LEVELS mip levels)
 *
 * Call `computeBRDFLUT` once after init (encodes a compute pass).
 * Call `convolve` whenever `scene.environment` changes.
 */
export class IBLPrefilter {
  readonly brdfLUT: GPUTexture;
  readonly brdfLUTView: GPUTextureView;
  readonly irradiance: GPUTexture;
  readonly irradianceView: GPUTextureView;
  readonly prefiltered: GPUTexture;
  readonly prefilteredView: GPUTextureView;
  readonly sampler: GPUSampler;

  private module: GPUShaderModule;
  private brdfLayout: GPUBindGroupLayout;
  private iblLayout: GPUBindGroupLayout;
  private brdfPipeline: GPUComputePipeline;
  private irrPipeline: GPUComputePipeline;
  private specPipeline: GPUComputePipeline;
  // Pre-filled per-dispatch param buffers (one for irradiance, one per mip level).
  private irrParams: GPUBuffer;
  private specParams: GPUBuffer[];
  private brdfDone = false;

  constructor(private device: GPUDevice) {
    this.module = device.createShaderModule({ code: IBL_SHADER, label: 'ibl' });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });

    this.brdfLayout = device.createBindGroupLayout({
      label: 'ibl-brdf',
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
      }],
    });

    this.iblLayout = device.createBindGroupLayout({
      label: 'ibl-env',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const brdfPL = device.createPipelineLayout({ bindGroupLayouts: [this.brdfLayout] });
    const iblPL  = device.createPipelineLayout({ bindGroupLayouts: [this.iblLayout] });

    this.brdfPipeline = device.createComputePipeline({ label: 'brdf-lut', layout: brdfPL, compute: { module: this.module, entryPoint: 'cs_brdf' } });
    this.irrPipeline  = device.createComputePipeline({ label: 'irradiance', layout: iblPL, compute: { module: this.module, entryPoint: 'cs_irradiance' } });
    this.specPipeline = device.createComputePipeline({ label: 'specular-prefilter', layout: iblPL, compute: { module: this.module, entryPoint: 'cs_specular' } });

    // Output textures.
    this.brdfLUT = device.createTexture({
      size: [128, 128], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.brdfLUTView = this.brdfLUT.createView();

    this.irradiance = device.createTexture({
      size: [64, 32], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.irradianceView = this.irradiance.createView();

    this.prefiltered = device.createTexture({
      size: [IBL_SPEC_W, IBL_SPEC_H], format: 'rgba16float',
      mipLevelCount: IBL_MIP_LEVELS,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.prefilteredView = this.prefiltered.createView();

    // Pre-fill param buffers (all values are compile-time constants).
    this.irrParams = this.makeParams([0, 0, 64, 32]);
    this.specParams = [];
    for (let mip = 0; mip < IBL_MIP_LEVELS; mip++) {
      const w = IBL_SPEC_W >> mip;
      const h = IBL_SPEC_H >> mip;
      const roughness = IBL_MIP_LEVELS <= 1 ? 0 : mip / (IBL_MIP_LEVELS - 1);
      this.specParams.push(this.makeParams([roughness, 0, w, h]));
    }
  }

  private makeParams(data: [number, number, number, number]): GPUBuffer {
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, new Float32Array(data));
    return buf;
  }

  /** Compute the BRDF LUT.  Call once after init (idempotent). */
  computeBRDFLUT(encoder: GPUCommandEncoder): void {
    if (this.brdfDone) return;
    const bg = this.device.createBindGroup({
      layout: this.brdfLayout,
      entries: [{ binding: 0, resource: this.brdfLUTView }],
    });
    const pass = encoder.beginComputePass({ label: 'brdf-lut' });
    pass.setPipeline(this.brdfPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(128 / 8), Math.ceil(128 / 8));
    pass.end();
    this.brdfDone = true;
  }

  /**
   * Convolve `envView` (equirectangular, sampleType 'float') into the irradiance
   * and prefiltered specular textures.  Call when `scene.environment` changes.
   */
  convolve(encoder: GPUCommandEncoder, envView: GPUTextureView): void {
    // Irradiance
    const irrBG = this.device.createBindGroup({
      layout: this.iblLayout,
      entries: [
        { binding: 0, resource: envView },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.irradianceView },
        { binding: 3, resource: { buffer: this.irrParams } },
      ],
    });
    const irrPass = encoder.beginComputePass({ label: 'irradiance' });
    irrPass.setPipeline(this.irrPipeline);
    irrPass.setBindGroup(0, irrBG);
    irrPass.dispatchWorkgroups(Math.ceil(64 / 8), Math.ceil(32 / 8));
    irrPass.end();

    // Specular prefilter: one dispatch per mip level.
    for (let mip = 0; mip < IBL_MIP_LEVELS; mip++) {
      const w = IBL_SPEC_W >> mip;
      const h = IBL_SPEC_H >> mip;
      const mipView = this.prefiltered.createView({ baseMipLevel: mip, mipLevelCount: 1 });
      const bg = this.device.createBindGroup({
        layout: this.iblLayout,
        entries: [
          { binding: 0, resource: envView },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: mipView },
          { binding: 3, resource: { buffer: this.specParams[mip] } },
        ],
      });
      const pass = encoder.beginComputePass({ label: `specular-mip${mip}` });
      pass.setPipeline(this.specPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(w / 8)), Math.max(1, Math.ceil(h / 8)));
      pass.end();
    }
  }
}
