export type TypedArray =
  | Float32Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Int16Array
  | Int8Array;

/** A view over interleaved or tightly-packed vertex data for one attribute. */
export class BufferAttribute {
  array: TypedArray;
  itemSize: number;
  normalized: boolean;
  /** Bumped when the data changes so the renderer can re-upload. */
  version = 0;

  constructor(array: TypedArray, itemSize: number, normalized = false) {
    this.array = array;
    this.itemSize = itemSize;
    this.normalized = normalized;
  }

  get count(): number {
    return this.array.length / this.itemSize;
  }

  getX(i: number): number {
    return this.array[i * this.itemSize];
  }
  getY(i: number): number {
    return this.array[i * this.itemSize + 1];
  }
  getZ(i: number): number {
    return this.array[i * this.itemSize + 2];
  }

  needsUpdate(): void {
    this.version++;
  }
}
