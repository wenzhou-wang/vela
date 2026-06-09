import { GLTFLoader } from './GLTFLoader';
import type { GLTFResult } from './GLTFLoader';
import type { GLTFRoot } from './gltfTypes';
import type { MeshoptDecoder, DracoDecoder } from './decoders';
import type { KTX2TextureLoader } from './decoders';

/**
 * A drop-in replacement for {@link GLTFLoader} that performs the I/O-heavy work
 * (buffer fetching, image decoding via `createImageBitmap`) in a Web Worker,
 * keeping the main thread free during loads. The final scene-graph construction
 * (geometry, materials, Object3D hierarchy) still runs on the main thread.
 *
 * Decoder plugins (meshopt, Draco, KTX2) are forwarded to the underlying
 * {@link GLTFLoader} and applied on the main thread after the worker returns.
 *
 * Usage:
 * ```ts
 * const loader = new WorkerGLTFLoader();
 * const { scene, animations } = await loader.load('/model.glb');
 * ```
 *
 * Call {@link terminate} when the loader is no longer needed to release the worker.
 */
export class WorkerGLTFLoader {
  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (v: GLTFResult) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private loader = new GLTFLoader();

  setMeshoptDecoder(decoder: MeshoptDecoder): this {
    this.loader.setMeshoptDecoder(decoder);
    return this;
  }

  setDracoDecoder(decoder: DracoDecoder): this {
    this.loader.setDracoDecoder(decoder);
    return this;
  }

  setKTX2Loader(loader: KTX2TextureLoader): this {
    this.loader.setKTX2Loader(loader);
    return this;
  }

  async load(url: string): Promise<GLTFResult> {
    const worker = this.getWorker();
    return new Promise<GLTFResult>((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ id, url });
    });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const { reject } of this.pending.values()) {
      reject(new Error('[vela] WorkerGLTFLoader terminated'));
    }
    this.pending.clear();
  }

  private getWorker(): Worker {
    if (!this.worker) {
      const blob = new Blob([WORKER_CODE], { type: 'text/javascript' });
      const url = URL.createObjectURL(blob);
      this.worker = new Worker(url);
      URL.revokeObjectURL(url);
      this.worker.onmessage = (e: MessageEvent) => void this.onMessage(e);
      this.worker.onerror = (e: ErrorEvent) => {
        for (const { reject } of this.pending.values()) reject(new Error(e.message));
        this.pending.clear();
      };
    }
    return this.worker;
  }

  private async onMessage(e: MessageEvent): Promise<void> {
    const { id, error, json, buffers, images } = e.data as {
      id: number;
      error?: string;
      json?: GLTFRoot;
      buffers?: ArrayBuffer[];
      images?: (ImageBitmap | null)[];
    };
    const p = this.pending.get(id);
    if (!p) return;
    this.pending.delete(id);

    if (error) {
      p.reject(new Error(`[vela] WorkerGLTFLoader: ${error}`));
      return;
    }
    try {
      const result = await this.loader.buildFromPreloaded(json!, buffers!, images!);
      p.resolve(result);
    } catch (err) {
      p.reject(err instanceof Error ? err : new Error(String(err)));
    }
  }
}

// ---------------------------------------------------------------------------
// Inline worker code
// ---------------------------------------------------------------------------

const WORKER_CODE = /* javascript */ `
'use strict';

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN  = 0x004e4942;

function parseGLB(data) {
  const dv  = new DataView(data);
  const len = dv.getUint32(8, true);
  let   off = 12, json = null, binary = null;
  while (off < len) {
    const chunkLen  = dv.getUint32(off,     true);
    const chunkType = dv.getUint32(off + 4, true);
    if (chunkType === CHUNK_JSON) {
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, off + 8, chunkLen)));
    } else if (chunkType === CHUNK_BIN) {
      binary = data.slice(off + 8, off + 8 + chunkLen); // copy to own the buffer
    }
    off += 8 + chunkLen;
  }
  return { json, binary };
}

function decodeDataURI(uri) {
  const comma = uri.indexOf(',');
  const meta  = uri.substring(5, comma);
  const data  = uri.substring(comma + 1);
  if (meta.includes('base64')) {
    const bin   = atob(data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }
  return new TextEncoder().encode(decodeURIComponent(data)).buffer;
}

async function loadBuffers(json, baseUrl, glbBinary) {
  const out = [];
  for (const buf of json.buffers ?? []) {
    if (buf.uri === undefined) {
      out.push(glbBinary ?? new ArrayBuffer(0));
    } else if (buf.uri.startsWith('data:')) {
      out.push(decodeDataURI(buf.uri));
    } else {
      const res = await fetch(baseUrl + decodeURIComponent(buf.uri));
      out.push(await res.arrayBuffer());
    }
  }
  return out;
}

async function loadImages(json, baseUrl, buffers) {
  const out = [];
  for (const image of json.images ?? []) {
    let blob;
    try {
      if (image.uri && image.uri.startsWith('data:')) {
        blob = new Blob([decodeDataURI(image.uri)]);
      } else if (image.uri) {
        const res = await fetch(baseUrl + decodeURIComponent(image.uri));
        blob = await res.blob();
      } else if (image.bufferView !== undefined) {
        const bv  = json.bufferViews[image.bufferView];
        const buf = buffers[bv.buffer];
        const off = bv.byteOffset ?? 0;
        blob = new Blob([buf.slice(off, off + bv.byteLength)], { type: image.mimeType ?? 'image/png' });
      } else {
        out.push(null);
        continue;
      }
      out.push(await createImageBitmap(blob, { colorSpaceConversion: 'none' }));
    } catch (_) {
      out.push(null); // KTX2 and unknown types fall back to null; main thread handles them
    }
  }
  return out;
}

self.onmessage = async function(e) {
  const { id, url } = e.data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('HTTP ' + res.status + ' fetching ' + url);
    const data    = await res.arrayBuffer();
    const baseUrl = url.substring(0, url.lastIndexOf('/') + 1);

    const dv = new DataView(data);
    let json, glbBinary;
    if (data.byteLength >= 12 && dv.getUint32(0, true) === GLB_MAGIC) {
      const r = parseGLB(data);
      json      = r.json;
      glbBinary = r.binary;
    } else {
      json      = JSON.parse(new TextDecoder().decode(data));
      glbBinary = null;
    }

    const buffers  = await loadBuffers(json, baseUrl, glbBinary);
    const images   = await loadImages(json, baseUrl, buffers);

    const transferables = [];
    for (const ab of buffers) if (ab.byteLength > 0) transferables.push(ab);
    for (const img of images) if (img) transferables.push(img);

    self.postMessage({ id, json, buffers, images }, transferables);
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};
`;
