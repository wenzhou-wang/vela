export const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

/** Vertex attribute shader locations (must match pbr.wgsl). */
export const ATTRIB_LOCATION = {
  position: 0,
  normal: 1,
  uv: 2,
  tangent: 3,
} as const;

// Base geometry streams shared by every variant (slots/locations 0-3).
const BASE_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
  { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
  { arrayStride: 8, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }] },
  { arrayStride: 16, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x4' }] },
];

const colorStream = (location: number): GPUVertexBufferLayout => ({
  arrayStride: 16,
  attributes: [{ shaderLocation: location, offset: 0, format: 'float32x4' }],
});

/** Static/instanced/morph layout: base streams + per-vertex color (slot 4). */
export const VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...BASE_VERTEX_BUFFER_LAYOUT,
  colorStream(4),
];

/** Base + joints (uint32x4) + weights (float32x4) + per-vertex color (slot 6). */
export const SKINNED_VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  ...BASE_VERTEX_BUFFER_LAYOUT,
  { arrayStride: 16, attributes: [{ shaderLocation: 4, offset: 0, format: 'uint32x4' }] },
  { arrayStride: 16, attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x4' }] },
  colorStream(6),
];
