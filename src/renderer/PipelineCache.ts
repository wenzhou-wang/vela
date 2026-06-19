import type { StandardMaterial } from '../materials/StandardMaterial';
import type { LineBasicMaterial } from '../materials/LineBasicMaterial';
import { DEPTH_FORMAT, SHADOW_DEPTH_FORMAT, OIT_ACCUM_FORMAT, OIT_REVEAL_FORMAT, VERTEX_BUFFER_LAYOUT, SKINNED_VERTEX_BUFFER_LAYOUT } from './constants';
import { LINE_VERTEX_BUFFER_LAYOUT } from './shaders/line.wgsl';
import { SHADOW_VERTEX_BUFFER_LAYOUT } from './shaders/shadow.wgsl';
import { SKY_SHADER } from './shaders/sky.wgsl';
import { PARTICLE_DRAW_SHADER } from './shaders/particles.wgsl';
import { SPRITE_SHADER } from './shaders/sprite.wgsl';

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
  /** Skybox pass layout: frame uniform + raw equirect env texture/sampler. */
  readonly skyLayout: GPUBindGroupLayout;
  private skyModule: GPUShaderModule | null = null;
  /** Particle draw layout (group 1): particle pool + draw params. */
  readonly particleLayout: GPUBindGroupLayout;
  private particleModule: GPUShaderModule | null = null;
  /** Sprite/text batch layout (group 1): instances + batch params + texture. */
  readonly spriteLayout: GPUBindGroupLayout;
  private spriteModule: GPUShaderModule | null = null;
  private layouts: Record<PipelineVariant, GPUPipelineLayout>;
  // ShaderMaterial group-2 bind layouts keyed by texture count, and pipeline
  // layouts keyed by `${variant}|${textureCount}`.
  private customBindLayouts = new Map<number, GPUBindGroupLayout>();
  private customPipelineLayouts = new Map<string, GPUPipelineLayout>();
  private modules: Record<PipelineVariant, GPUShaderModule>;
  private customModules = new Map<string, GPUShaderModule>();
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
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        // IBL: irradiance map (6-7) and BRDF split-sum LUT (8-9).
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        // Spot/point shadow atlas (10-12).
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d' } },
        { binding: 12, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        // Screen-space refraction capture (13).
        { binding: 13, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        // Clustered forward+ light lists (14).
        { binding: 14, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.modelLayout = device.createBindGroupLayout({
      label: 'model',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });

    // material uniform + 7 (texture, sampler) pairs
    // 0: uniform, 1-10: base/normal/mr/emissive/occlusion, 11-14: clearcoatMap/clearcoatRoughnessMap
    const materialEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ];
    for (let i = 0; i < 7; i++) {
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

    // Skybox pass: frame uniform + raw env (frame binding 4 may hold the
    // low-res IBL-prefiltered map instead, so the sky binds the raw texture).
    this.skyLayout = device.createBindGroupLayout({
      label: 'sky',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    // Particle draw: group 0 reuses the frame layout; group 1 is pool + params.
    this.particleLayout = device.createBindGroupLayout({
      label: 'particles',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    // Sprite/text batches: instances + params + texture/sampler at group 1.
    this.spriteLayout = device.createBindGroupLayout({
      label: 'sprites',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

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
  getLine(material: LineBasicMaterial, format: GPUTextureFormat = this.format): GPURenderPipeline {
    const blend = material.transparent ? 'blend' : 'opaque';
    const depthWrite = material.depthWrite && !material.transparent ? 'dw1' : 'dw0';
    const depthTest = material.depthTest ? 'dt1' : 'dt0';
    const key = `line|${blend}|${depthWrite}|${depthTest}|${format}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const target: GPUColorTargetState = { format };
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

  /**
   * Weighted-blended OIT pipeline for a transparent material: two targets (accum
   * additive, revealage multiplicative), depth-tested but not depth-written.
   * `sc` is the multisample count (1 for non-MSAA, or the renderer's sampleCount).
   */
  getOIT(material: StandardMaterial, variant: PipelineVariant = 'static', sc = 1): GPURenderPipeline {
    const cull = material.side === 'double' ? 'none' : material.side === 'back' ? 'front' : 'back';
    const key = `oit|${variant}|${cull}|sc${sc}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const accum: GPUColorTargetState = {
      format: OIT_ACCUM_FORMAT,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      },
    };
    const reveal: GPUColorTargetState = {
      format: OIT_REVEAL_FORMAT,
      blend: {
        color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
        alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
      },
    };

    const module = this.modules[variant];
    pipeline = this.device.createRenderPipeline({
      label: `pbr-${key}`,
      layout: this.layouts[variant],
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: variant === 'skinned' ? SKINNED_VERTEX_BUFFER_LAYOUT : VERTEX_BUFFER_LAYOUT,
      },
      fragment: { module, entryPoint: 'fs_oit', targets: [accum, reveal] },
      primitive: { topology: 'triangle-list', cullMode: cull as GPUCullMode, frontFace: 'ccw' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: sc },
    });
    this.cache.set(key, pipeline);
    return pipeline;
  }

  /**
   * Group-2 bind layout for a ShaderMaterial: a uniform buffer at binding 0
   * (when it has scalar uniforms) plus `textureCount` texture+sampler pairs at
   * bindings 1,2 / 3,4 / … — exactly matching the generated WGSL. Cached per
   * `(hasBuffer, textureCount)` shape.
   */
  customUniformLayout(hasBuffer: boolean, textureCount: number): GPUBindGroupLayout {
    const shape = (hasBuffer ? 1 : 0) * 1000 + textureCount;
    let layout = this.customBindLayouts.get(shape);
    if (layout) return layout;
    const entries: GPUBindGroupLayoutEntry[] = [];
    if (hasBuffer) {
      entries.push({ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
    }
    for (let i = 0; i < textureCount; i++) {
      entries.push({ binding: 1 + i * 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } });
      entries.push({ binding: 2 + i * 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } });
    }
    layout = this.device.createBindGroupLayout({ label: `shader-material-${shape}`, entries });
    this.customBindLayouts.set(shape, layout);
    return layout;
  }

  private customPipelineLayout(variant: PipelineVariant, hasBuffer: boolean, textureCount: number): GPUPipelineLayout {
    const key = `${variant}|${hasBuffer ? 1 : 0}|${textureCount}`;
    let layout = this.customPipelineLayouts.get(key);
    if (layout) return layout;
    const group2 = this.customUniformLayout(hasBuffer, textureCount);
    const groups =
      variant === 'instanced' ? [this.frameLayout, this.instanceLayout, group2]
      : variant === 'skinned' ? [this.frameLayout, this.modelLayout, group2, this.bonesLayout]
      : variant === 'morph' ? [this.frameLayout, this.modelLayout, group2, this.morphLayout]
      : [this.frameLayout, this.modelLayout, group2];
    layout = this.device.createPipelineLayout({ bindGroupLayouts: groups });
    this.customPipelineLayouts.set(key, layout);
    return layout;
  }

  /**
   * Pipeline for a ShaderMaterial. `cacheKey` must change whenever the
   * generated WGSL would (material id + version + uniform shape); `buildCode`
   * is only invoked when the module isn't cached yet. Pass the actual scene
   * color format (`rgba16float` under post-processing). `hasBuffer`/`textureCount`
   * describe the group-2 bind layout the generated WGSL expects.
   */
  getCustom(
    cacheKey: string,
    variant: PipelineVariant,
    material: { side: string; transparent: boolean; depthWrite: boolean },
    format: GPUTextureFormat,
    oit: boolean,
    oitSampleCount: number,
    buildCode: () => string,
    hasBuffer: boolean,
    textureCount: number,
    label = 'ShaderMaterial',
    cullOverride?: GPUCullMode,
  ): GPURenderPipeline {
    const pipelineLayout = this.customPipelineLayout(variant, hasBuffer, textureCount);
    const cull = cullOverride ?? (material.side === 'double' ? 'none' : material.side === 'back' ? 'front' : 'back');
    const blend = material.transparent ? 'blend' : 'opaque';
    const depthWrite = material.depthWrite && !material.transparent ? 'dw1' : 'dw0';
    const key = oit
      ? `cust|${cacheKey}|${variant}|${cull}|oit|sc${oitSampleCount}`
      : `cust|${cacheKey}|${variant}|${cull}|${blend}|${depthWrite}|${format}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const moduleKey = `${cacheKey}|${variant}`;
    let module = this.customModules.get(moduleKey);
    if (!module) {
      // Evict modules/pipelines from older versions of this material (the
      // cacheKey prefix up to the version segment identifies the material).
      const prefix = cacheKey.slice(0, cacheKey.indexOf(':v') + 2);
      for (const k of this.customModules.keys()) {
        if (k.startsWith(prefix) && !k.startsWith(cacheKey)) this.customModules.delete(k);
      }
      for (const k of this.cache.keys()) {
        if (k.startsWith(`cust|${prefix}`) && !k.startsWith(`cust|${cacheKey}`)) this.cache.delete(k);
      }
      const code = buildCode();
      module = this.device.createShaderModule({ code, label: `shader-material:${label}` });
      this.logCompilationErrors(module, code, label);
      this.customModules.set(moduleKey, module);
    }

    if (oit) {
      const accum: GPUColorTargetState = {
        format: OIT_ACCUM_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      };
      const reveal: GPUColorTargetState = {
        format: OIT_REVEAL_FORMAT,
        blend: {
          color: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
          alpha: { srcFactor: 'zero', dstFactor: 'one-minus-src', operation: 'add' },
        },
      };
      pipeline = this.device.createRenderPipeline({
        label: key,
        layout: pipelineLayout,
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: variant === 'skinned' ? SKINNED_VERTEX_BUFFER_LAYOUT : VERTEX_BUFFER_LAYOUT,
        },
        fragment: { module, entryPoint: 'fs_oit', targets: [accum, reveal] },
        primitive: { topology: 'triangle-list', cullMode: cull as GPUCullMode, frontFace: 'ccw' },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
        multisample: { count: oitSampleCount },
      });
    } else {
      const target: GPUColorTargetState = { format };
      if (blend === 'blend') {
        target.blend = {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        };
      }
      pipeline = this.device.createRenderPipeline({
        label: key,
        layout: pipelineLayout,
        vertex: {
          module,
          entryPoint: 'vs_main',
          buffers: variant === 'skinned' ? SKINNED_VERTEX_BUFFER_LAYOUT : VERTEX_BUFFER_LAYOUT,
        },
        fragment: { module, entryPoint: 'fs_main', targets: [target] },
        primitive: { topology: 'triangle-list', cullMode: cull as GPUCullMode, frontFace: 'ccw' },
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: depthWrite === 'dw1',
          depthCompare: 'less',
        },
        multisample: { count: this.sampleCount },
      });
    }
    this.cache.set(key, pipeline);
    return pipeline;
  }

  /**
   * Skybox pipeline: fullscreen triangle at depth 1, depth-tested (less-equal)
   * but not written, drawn inside the scene pass after the opaques.
   */
  getSky(format: GPUTextureFormat): GPURenderPipeline {
    const key = `sky|${format}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    this.skyModule ??= this.device.createShaderModule({ code: SKY_SHADER, label: 'sky' });
    pipeline = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.skyLayout] }),
      vertex: { module: this.skyModule, entryPoint: 'vs_sky' },
      fragment: { module: this.skyModule, entryPoint: 'fs_sky', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: this.sampleCount },
    });
    this.cache.set(key, pipeline);
    return pipeline;
  }

  /**
   * Particle billboard pipeline: 6 vertices per instance, depth-tested but not
   * written, premultiplied blending — additive (one,one) or alpha
   * (one, one-minus-src-alpha).
   */
  getParticles(format: GPUTextureFormat, additive: boolean): GPURenderPipeline {
    const key = `particles|${format}|${additive ? 'add' : 'alpha'}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    this.particleModule ??= this.device.createShaderModule({ code: PARTICLE_DRAW_SHADER, label: 'particles' });
    const blend: GPUBlendState = additive
      ? {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        }
      : {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        };
    pipeline = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.particleLayout] }),
      vertex: { module: this.particleModule, entryPoint: 'vs_main' },
      fragment: { module: this.particleModule, entryPoint: 'fs_main', targets: [{ format, blend }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });
    this.cache.set(key, pipeline);
    return pipeline;
  }

  /**
   * Sprite/SDF-text pipeline: premultiplied alpha blending; world-space
   * batches are depth-tested (no write), screen-space batches skip the depth
   * test entirely (HUD overlay).
   */
  getSprites(format: GPUTextureFormat, screen: boolean): GPURenderPipeline {
    const key = `sprites|${format}|${screen ? 'screen' : 'world'}`;
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    this.spriteModule ??= this.device.createShaderModule({ code: SPRITE_SHADER, label: 'sprites' });
    pipeline = this.device.createRenderPipeline({
      label: key,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.spriteLayout] }),
      vertex: { module: this.spriteModule, entryPoint: 'vs_main' },
      fragment: {
        module: this.spriteModule,
        entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: screen ? 'always' : 'less',
      },
      multisample: { count: this.sampleCount },
    });
    this.cache.set(key, pipeline);
    return pipeline;
  }

  /** Print WGSL compile errors with the offending source lines (async, fire-and-forget). */
  private logCompilationErrors(module: GPUShaderModule, code: string, label: string): void {
    module.getCompilationInfo().then((info) => {
      const problems = info.messages.filter((m) => m.type === 'error');
      if (problems.length === 0) return;
      const lines = code.split('\n');
      let report = `[vela] ShaderMaterial "${label}" failed to compile:\n`;
      for (const m of problems) {
        report += `  :${m.lineNum}:${m.linePos} ${m.message}\n`;
        const src = lines[m.lineNum - 1];
        if (src !== undefined) report += `    ${m.lineNum} | ${src}\n`;
      }
      report += 'Fix the `surface` WGSL (signature: fn surface(in : VSOut) -> Surface).';
      console.error(report);
    });
  }

  private keyFor(material: StandardMaterial, variant: PipelineVariant, format: GPUTextureFormat, cullOverride?: GPUCullMode): string {
    const cull = cullOverride ?? (material.side === 'double' ? 'none' : material.side === 'back' ? 'front' : 'back');
    const blend = material.transparent ? 'blend' : 'opaque';
    const depthWrite = material.depthWrite && !material.transparent ? 'dw1' : 'dw0';
    return `${variant}|${cull}|${blend}|${depthWrite}|${format}`;
  }

  /**
   * Compile/cache the opaque/transparent pipeline for a StandardMaterial.
   * `cullOverride` forces a cull mode regardless of `material.side` — used by
   * the shell pass to draw back faces ('front' cull) of the extruded hull.
   */
  get(material: StandardMaterial, variant: PipelineVariant = 'static', format: GPUTextureFormat = this.format, cullOverride?: GPUCullMode): GPURenderPipeline {
    const key = this.keyFor(material, variant, format, cullOverride);
    let pipeline = this.cache.get(key);
    if (pipeline) return pipeline;

    const [, cull, blend, depthWrite] = key.split('|');

    const target: GPUColorTargetState = { format };
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
