import { computeUniformLayout, packUniforms, type UniformValue, type UniformLayout } from '../materials/ShaderMaterial';

/** A storage buffer binding: either a byte size, or initial data to upload. */
export interface StorageBinding {
  kind: 'storage';
  /** Byte size; inferred from `data` when omitted. */
  size?: number;
  /** Initial contents (also sets size). */
  data?: ArrayBufferView<ArrayBuffer>;
  /** Allow CPU read-back via `task.read(name)`. Default false. */
  readable?: boolean;
}

/** A uniform buffer binding holding auto-packed scalar/vector values. */
export interface UniformBinding {
  kind: 'uniform';
  values: Record<string, UniformValue>;
}

export type ComputeBinding = StorageBinding | UniformBinding;

/** `storage({...})` / `uniform({...})` helpers for `ComputeTask` bindings. */
export function storage(opts: Omit<StorageBinding, 'kind'> = {}): StorageBinding {
  return { kind: 'storage', ...opts };
}
export function uniform(values: Record<string, UniformValue>): UniformBinding {
  return { kind: 'uniform', values };
}

export interface ComputeTaskOptions {
  /**
   * WGSL compute source. Bind the declared buffers at `@group(0)` in
   * declaration order (binding 0, 1, 2, …) and define an `@compute` entry
   * point (default name `main`). Texture bindings are not supported here —
   * use storage/uniform buffers.
   */
  code: string;
  /** Compute entry point. Default `'main'`. */
  entryPoint?: string;
  /**
   * Named bindings in `@group(0)` order. Storage buffers are `var<storage,
   * read_write>`; uniform buffers are `var<uniform>` with the packed layout
   * (`computeUniformLayout`). Bindings are assigned 0,1,2,… in this order.
   */
  bindings: Record<string, ComputeBinding>;
  /** Default dispatch size (workgroup counts); overridable per `dispatch()`. */
  workgroups?: [number, number?, number?];
  name?: string;
}

/**
 * A declarative GPU compute job: WGSL plus named storage/uniform buffers, with
 * auto-built bind group layout and uniform packing — no raw WebGPU plumbing.
 * Run it standalone (`await task.run()`) or fold it into a frame by passing the
 * frame's encoder. Read storage results back with `task.read(name)` (requires
 * `readable: true`).
 *
 * ```ts
 * import { ComputeTask, storage, uniform } from 'vela';
 *
 * const N = 1024;
 * const task = new ComputeTask({
 *   bindings: {
 *     data: storage({ size: N * 4, readable: true }),
 *     params: uniform({ scale: 2.0 }),
 *   },
 *   workgroups: [Math.ceil(N / 64)],
 *   code: `
 *     @group(0) @binding(0) var<storage, read_write> data : array<f32>;
 *     struct P { scale : f32 };
 *     @group(0) @binding(1) var<uniform> params : P;
 *     @compute @workgroup_size(64)
 *     fn main(@builtin(global_invocation_id) id : vec3<u32>) {
 *       data[id.x] = f32(id.x) * params.scale;
 *     }`,
 * });
 * await task.init(device);
 * const out = new Float32Array(await task.read('data'));
 * ```
 *
 * When used with a renderer, `task.init(renderer.device)` shares the GPU device.
 */
export class ComputeTask {
  readonly name: string;
  private readonly options: ComputeTaskOptions;
  private device: GPUDevice | null = null;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private order: string[] = [];
  private buffers = new Map<string, GPUBuffer>();
  private readback = new Map<string, GPUBuffer>();
  private uniformLayouts = new Map<string, UniformLayout>();

  constructor(options: ComputeTaskOptions) {
    if (!options.code || !/(@compute)/.test(options.code)) {
      throw new Error('ComputeTask: `code` must be WGSL containing an @compute entry point.');
    }
    if (!options.bindings || Object.keys(options.bindings).length === 0) {
      throw new Error('ComputeTask: provide at least one binding (storage/uniform).');
    }
    this.options = options;
    this.name = options.name ?? 'ComputeTask';
  }

  /** Create GPU resources on `device`. Call once before dispatching. */
  async init(device: GPUDevice): Promise<void> {
    if (this.device) return;
    this.device = device;
    this.order = Object.keys(this.options.bindings);

    const layoutEntries: GPUBindGroupLayoutEntry[] = [];
    const groupEntries: GPUBindGroupEntry[] = [];
    this.order.forEach((name, binding) => {
      const b = this.options.bindings[name];
      if (b.kind === 'storage') {
        const size = b.size ?? b.data?.byteLength ?? 0;
        if (size <= 0) throw new Error(`ComputeTask binding "${name}": storage needs a size or data.`);
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
          (b.readable ? GPUBufferUsage.COPY_SRC : 0);
        const buf = device.createBuffer({ label: `${this.name}:${name}`, size, usage });
        if (b.data) device.queue.writeBuffer(buf, 0, b.data);
        this.buffers.set(name, buf);
        layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
        groupEntries.push({ binding, resource: { buffer: buf } });
      } else {
        const layout = computeUniformLayout(b.values, 0);
        this.uniformLayouts.set(name, layout);
        const buf = device.createBuffer({
          label: `${this.name}:${name}`,
          size: layout.size,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const data = new Float32Array(layout.size / 4);
        packUniforms(b.values, layout, data);
        device.queue.writeBuffer(buf, 0, data);
        this.buffers.set(name, buf);
        layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        groupEntries.push({ binding, resource: { buffer: buf } });
      }
    });

    const bindLayout = device.createBindGroupLayout({ label: this.name, entries: layoutEntries });
    const module = device.createShaderModule({ code: this.options.code, label: this.name });
    module.getCompilationInfo().then((info) => {
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length === 0) return;
      const lines = this.options.code.split('\n');
      let report = `[vela] ComputeTask "${this.name}" failed to compile:\n`;
      for (const m of errs) {
        report += `  :${m.lineNum}:${m.linePos} ${m.message}\n`;
        if (lines[m.lineNum - 1] !== undefined) report += `    ${m.lineNum} | ${lines[m.lineNum - 1]}\n`;
      }
      console.error(report);
    });
    this.pipeline = device.createComputePipeline({
      label: this.name,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
      compute: { module, entryPoint: this.options.entryPoint ?? 'main' },
    });
    this.bindGroup = device.createBindGroup({ layout: bindLayout, entries: groupEntries });
  }

  /**
   * Re-pack and upload a uniform binding's values (call after mutating them).
   * The value shape must match what was declared at construction.
   */
  updateUniform(name: string, values: Record<string, UniformValue>): void {
    const layout = this.uniformLayouts.get(name);
    const buf = this.buffers.get(name);
    if (!layout || !buf) throw new Error(`ComputeTask: "${name}" is not a uniform binding (call init first).`);
    const data = new Float32Array(layout.size / 4);
    packUniforms(values, layout, data);
    this.device!.queue.writeBuffer(buf, 0, data);
  }

  /** Overwrite a storage buffer's contents (offset in bytes). */
  write(name: string, data: ArrayBufferView<ArrayBuffer>, offset = 0): void {
    const buf = this.requireBuffer(name);
    this.device!.queue.writeBuffer(buf, offset, data);
  }

  /**
   * Encode the dispatch. With no encoder, runs standalone (own encoder +
   * submit); pass a frame encoder to fold it into a frame's command stream.
   */
  dispatch(encoder?: GPUCommandEncoder, workgroups?: [number, number?, number?]): void {
    if (!this.device || !this.pipeline || !this.bindGroup) {
      throw new Error('ComputeTask: call await task.init(device) before dispatch().');
    }
    const wg = workgroups ?? this.options.workgroups ?? [1];
    const own = encoder ?? this.device.createCommandEncoder();
    const pass = own.beginComputePass({ label: this.name });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(wg[0], wg[1] ?? 1, wg[2] ?? 1);
    pass.end();
    if (!encoder) this.device.queue.submit([own.finish()]);
  }

  /** Convenience: dispatch standalone. */
  run(workgroups?: [number, number?, number?]): void {
    this.dispatch(undefined, workgroups);
  }

  /**
   * Read a `readable` storage buffer back to the CPU (resolves after the GPU
   * finishes). Returns a copy as an ArrayBuffer.
   */
  async read(name: string): Promise<ArrayBuffer> {
    const buf = this.requireBuffer(name);
    const binding = this.options.bindings[name];
    if (binding.kind !== 'storage' || !binding.readable) {
      throw new Error(`ComputeTask: binding "${name}" must be storage({ readable: true }) to read back.`);
    }
    const size = buf.size;
    let staging = this.readback.get(name);
    if (!staging) {
      staging = this.device!.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.readback.set(name, staging);
    }
    const encoder = this.device!.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, staging, 0, size);
    this.device!.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return copy;
  }

  /** Raw GPU buffer for a binding (e.g. to bind elsewhere). */
  buffer(name: string): GPUBuffer {
    return this.requireBuffer(name);
  }

  /** Release all GPU buffers. */
  dispose(): void {
    for (const b of this.buffers.values()) b.destroy();
    for (const b of this.readback.values()) b.destroy();
    this.buffers.clear();
    this.readback.clear();
  }

  private requireBuffer(name: string): GPUBuffer {
    const buf = this.buffers.get(name);
    if (!buf) throw new Error(`ComputeTask: no binding "${name}" (or init() not called).`);
    return buf;
  }
}
