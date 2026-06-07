import type { StandardMaterial } from '../materials/StandardMaterial';
import { DEPTH_FORMAT, VERTEX_BUFFER_LAYOUT, SKINNED_VERTEX_BUFFER_LAYOUT } from './constants';

/**
 * Owns the shared bind group layouts and compiles/caches render pipelines.
 * Because the shader is a single uber-shader, pipelines only vary by render
 * state (cull mode, blending, depth write) and whether the mesh is skinned.
 */
export type PipelineVariant = 'static' | 'skinned' | 'instanced';

export class PipelineCache {
  readonly frameLayout: GPUBindGroupLayout;
  readonly modelLayout: GPUBindGroupLayout;
  readonly materialLayout: GPUBindGroupLayout;
  readonly bonesLayout: GPUBindGroupLayout;
  readonly instanceLayout: GPUBindGroupLayout;
  private layouts: Record<PipelineVariant, GPUPipelineLayout>;
  private modules: Record<PipelineVariant, GPUShaderModule>;
  private cache = new Map<string, GPURenderPipeline>();

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private sampleCount: number,
    staticCode: string,
    skinnedCode: string,
    instancedCode: string,
  ) {
    this.modules = {
      static: device.createShaderModule({ code: staticCode, label: 'pbr' }),
      skinned: device.createShaderModule({ code: skinnedCode, label: 'pbr-skinned' }),
      instanced: device.createShaderModule({ code: instancedCode, label: 'pbr-instanced' }),
    };

    this.frameLayout = device.createBindGroupLayout({
      label: 'frame',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.modelLayout = device.createBindGroupLayout({
      label: 'model',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    // material uniform + 5 (texture, sampler) pairs
    const materialEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ];
    for (let i = 0; i < 5; i++) {
      materialEntries.push({
        binding: 1 + i * 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d' },
      });
      materialEntries.push({
        binding: 2 + i * 2,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' },
      });
    }
    this.materialLayout = device.createBindGroupLayout({ label: 'material', entries: materialEntries });

    this.bonesLayout = device.createBindGroupLayout({
      label: 'bones',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    // Per-instance model matrices, replacing the model uniform at group 1.
    this.instanceLayout = device.createBindGroupLayout({
      label: 'instances',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.layouts = {
      static: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.modelLayout, this.materialLayout],
      }),
      skinned: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.modelLayout, this.materialLayout, this.bonesLayout],
      }),
      instanced: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.instanceLayout, this.materialLayout],
      }),
    };
  }

  private keyFor(material: StandardMaterial, variant: PipelineVariant): string {
    const cull = material.side === 'double' ? 'none' : material.side === 'back' ? 'front' : 'back';
    const blend = material.transparent ? 'blend' : 'opaque';
    const depthWrite = material.depthWrite && !material.transparent ? 'dw1' : 'dw0';
    return `${variant}|${cull}|${blend}|${depthWrite}`;
  }

  get(material: StandardMaterial, variant: PipelineVariant = 'static'): GPURenderPipeline {
    const key = this.keyFor(material, variant);
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const [, cull, blend, depthWrite] = key.split('|');

    const target: GPUColorTargetState = { format: this.format };
    if (blend === 'blend') {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    }

    const module = this.modules[variant];
    pipeline = this.device.createRenderPipeline({
      label: `pbr-${key}`,
      layout: this.layouts[variant],
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: variant === 'skinned' ? SKINNED_VERTEX_BUFFER_LAYOUT : VERTEX_BUFFER_LAYOUT,
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [target],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: cull as GPUCullMode,
        frontFace: 'ccw',
      },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: depthWrite === 'dw1',
        depthCompare: 'less',
      },
      multisample: { count: this.sampleCount },
    });

    this.cache.set(key, pipeline);
    return pipeline;
  }
}
