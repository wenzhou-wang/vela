import type { BufferGeometry } from '../core/BufferGeometry';
import { BufferAttribute } from '../core/BufferAttribute';

interface GPUGeometry {
  position: GPUBuffer;
  normal: GPUBuffer;
  uv: GPUBuffer;
  tangent: GPUBuffer;
  /** Per-vertex vec4 color; defaults to opaque white when the geometry has none. */
  color: GPUBuffer;
  /** Skinning streams, present only for skinned geometries. */
  joints: GPUBuffer | null;
  weights: GPUBuffer | null;
  index: GPUBuffer | null;
  indexFormat: GPUIndexFormat;
  drawCount: number;
  version: number;
}

/**
 * Lazily creates and caches GPU vertex/index buffers for geometries.
 * Fills in default normal/uv/tangent buffers when a geometry lacks them so the
 * fixed vertex layout always has all four streams.
 */
export class GeometryBuffers {
  private cache = new WeakMap<BufferGeometry, GPUGeometry>();

  constructor(private device: GPUDevice) {}

  get(geometry: BufferGeometry): GPUGeometry {
    const existing = this.cache.get(geometry);
    if (existing && existing.version === geometry.version) return existing;
    if (existing) this.dispose(existing);

    const gpu = this.create(geometry);
    this.cache.set(geometry, gpu);
    return gpu;
  }

  private create(geometry: BufferGeometry): GPUGeometry {
    const pos = geometry.getAttribute('position');
    if (!pos) throw new Error('[vela] geometry has no position attribute');
    const vertexCount = pos.count;

    let normal = geometry.getAttribute('normal');
    if (!normal) {
      geometry.computeVertexNormals();
      normal = geometry.getAttribute('normal')!;
    }

    const uv = geometry.getAttribute('uv') ?? new BufferAttribute(new Float32Array(vertexCount * 2), 2);
    const tangent = geometry.getAttribute('tangent') ?? defaultTangents(vertexCount);

    const positionBuf = this.upload(pos.array as Float32Array, GPUBufferUsage.VERTEX);
    const normalBuf = this.upload(normal.array as Float32Array, GPUBufferUsage.VERTEX);
    const uvBuf = this.upload(uv.array as Float32Array, GPUBufferUsage.VERTEX);
    const tangentBuf = this.upload(tangent.array as Float32Array, GPUBufferUsage.VERTEX);

    // Per-vertex color (white default keeps the stream always present, so vertex
    // colors need no pipeline variant — absent colors multiply by 1).
    const colorAttr = geometry.getAttribute('color') ?? defaultColors(vertexCount);
    const color = this.upload(colorAttr.array as Float32Array, GPUBufferUsage.VERTEX);

    // Skinning streams (optional)
    const jointsAttr = geometry.getAttribute('joints');
    const weightsAttr = geometry.getAttribute('weights');
    const joints = jointsAttr ? this.upload(jointsAttr.array as Uint32Array, GPUBufferUsage.VERTEX) : null;
    const weights = weightsAttr ? this.upload(weightsAttr.array as Float32Array, GPUBufferUsage.VERTEX) : null;

    let index: GPUBuffer | null = null;
    let indexFormat: GPUIndexFormat = 'uint32';
    if (geometry.index) {
      let array = geometry.index.array;
      if (array instanceof Uint16Array) {
        indexFormat = 'uint16';
        // uint16 index buffers must be 4-byte aligned in size
        if ((array.byteLength & 3) !== 0) {
          const padded = new Uint16Array(array.length + 1);
          padded.set(array);
          array = padded;
        }
        index = this.upload(array, GPUBufferUsage.INDEX);
      } else {
        const u32 = array instanceof Uint32Array ? array : new Uint32Array(array);
        index = this.upload(u32, GPUBufferUsage.INDEX);
      }
    }

    return {
      position: positionBuf,
      normal: normalBuf,
      uv: uvBuf,
      tangent: tangentBuf,
      color,
      joints,
      weights,
      index,
      indexFormat,
      drawCount: geometry.getDrawCount(),
      version: geometry.version,
    };
  }

  private upload(data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
    const size = (data.byteLength + 3) & ~3; // round up to 4 bytes
    const buffer = this.device.createBuffer({
      size,
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data as ArrayBuffer & ArrayBufferView, 0);
    return buffer;
  }

  private dispose(gpu: GPUGeometry): void {
    gpu.position.destroy();
    gpu.normal.destroy();
    gpu.uv.destroy();
    gpu.tangent.destroy();
    gpu.color?.destroy();
    gpu.joints?.destroy();
    gpu.weights?.destroy();
    gpu.index?.destroy();
  }
}

function defaultTangents(count: number): BufferAttribute {
  const array = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    array[i * 4] = 1; // tangent along +X
    array[i * 4 + 3] = 1; // handedness
  }
  return new BufferAttribute(array, 4);
}

function defaultColors(count: number): BufferAttribute {
  return new BufferAttribute(new Float32Array(count * 4).fill(1), 4);
}
