import type { StandardMaterial } from '../materials/StandardMaterial';
import type { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { DEPTH_FORMAT, SHADOW_DEPTH_FORMAT, VERTEX_BUFFER_LAYOUT, SKINNED_VERTEX_BUFFER_LAYOUT } from './constants';
import { LINE_VERTEX_BUFFER_LAYOUT } from './shaders/line.wgsl';
import { SHADOW_VERTEX_BUFFER_LAYOUT } from './shaders/shadow.wgsl';

/**
 * Owns the shared bind group layouts and compiles/caches render pipelines.
 * Because the shader is a single uber-shader, pipelines only vary by render
 * state (cull mode, blending, depth write) and whether the mesh is skinned.
 */
export type PipelineVariant = 'static' | 'skinned' | 'instanced' | 'morph';

export class PipelineCache {
  readonly frameLayout: GPUBindGroupLayout;
  readonly modelLayout: GPUBindGroupLayout;
  readonly materialLayout: GPUBindGroupLayout;
  readonly bonesLayout: GPUBindGroupLayout;
  readonly instanceLayout: GPUBindGroupLayout;
  readonly morphLayout: GPUBindGroupLayout;
  readonly lineMaterialLayout: GPUBindGroupLayout;
  readonly shadowLightLayout: GPUBindGroupLayout;
  readonly shadowPipeline: GPURenderPipeline;
  private layouts: Record<PipelineVariant, GPUPipelineLayout>;
  private modules: Record<PipelineVariant, GPUShaderModule>;
  private lineLayout: GPUPipelineLayout;
  private lineModule: GPUShaderModule;
  private cache = new Map<string, GPURenderPipeline>();

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private sampleCount: number,
    staticCode: string,
    skinnedCode: string,
    instancedCode: string,
    morphCode: string,
    lineCode: string,
    shadowCode: string,
  ) {
    this.modules = {
      static: device.createShaderModule({ code: staticCode, label: 'pbr' }),
      skinned: device.createShaderModule({ code: skinnedCode, label: 'pbr-skinned' }),
      instanced: device.createShaderModule({ code: instancedCode, label: 'pbr-instanced' }),
      morph: device.createShaderModule({ code: morphCode, label: 'pbr-morph' }),
    };

    this.frameLayout = device.createBindGroupLayout({
      label: 'frame',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
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

    // Morph info uniform + position/normal delta + weight storage buffers.
    this.morphLayout = device.createBindGroupLayout({
      label: 'morph',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
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
      morph: device.createPipelineLayout({
        bindGroupLayouts: [this.frameLayout, this.modelLayout, this.materialLayout, this.morphLayout],
      }),
    };

    // Unlit line path: frame + model + a small line-material uniform.
    this.lineMaterialLayout = device.createBindGroupLayout({
      label: 'line-material',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.lineModule = device.createShaderModule({ code: lineCode, label: 'line' });
    this.lineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.frameLayout, this.modelLayout, this.lineMaterialLayout],
    });

    // Shadow depth pass: light-matrix uniform (group 0) + the shared model group.
    this.shadowLightLayout = device.createBindGroupLayout({
      label: 'shadow-light',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    const shadowModule = device.createShaderModule({ code: shadowCode, label: 'shadow' });
    this.shadowPipeline = device.createRenderPipeline({
      label: 'shadow-depth',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.shadowLightLayout, this.modelLayout] }),
      vertex: { module: shadowModule, entryPoint: 'vs_main', buffers: SHADOW_VERTEX_BUFFER_LAYOUT },
      primitive: { topology: 'triangle-list', cullMode: 'none', frontFace: 'ccw' },
      depthStencil: { format: SHADOW_DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  /** Compile/cache the `line-list` pipeline for a line material's render state. */
  getLine(material: LineBasicMaterial): GPURenderPipeline {
    const blend = material.transparent ? 'blend' : 'opaque';
    const depthWrite = material.depthWrite && !material.transparent ? 'dw1' : 'dw0';
    const depthTest = material.depthTest ? 'dt1' : 'dt0';
    const key = `line|${blend}|${depthWrite}|${depthTest}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const target: GPUColorTargetState = { format: this.format };
    if (blend === 'blend') {
      target.blend = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      };
    }

    pipeline = this.device.createRenderPipeline({
      label: key,
      layout: this.lineLayout,
      vertex: { module: this.lineModule, entryPoint: 'vs_main', buffers: LINE_VERTEX_BUFFER_LAYOUT },
      fragment: { module: this.lineModule, entryPoint: 'fs_main', targets: [target] },
      primitive: { topology: 'line-list' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: depthWrite === 'dw1',
        depthCompare: depthTest === 'dt1' ? 'less' : 'always',
      },
      multisample: { count: this.sampleCount },
    });
    this.cache.set(key, pipeline);
    return pipeline;
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
