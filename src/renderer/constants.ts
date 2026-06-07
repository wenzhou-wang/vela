export const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

/** Vertex attribute shader locations (must match pbr.wgsl). */
export const ATTRIB_LOCATION = {
  position: 0,
  normal: 1,
  uv: 2,
  tangent: 3,
} as const;

export const VERTEX_BUFFER_LAYOUT: GPUVertexBufferLayout[] = [
  {
    arrayStride: 12,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
  },
  {
    arrayStride: 12,
    attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
  },
  {
    arrayStride: 8,
    attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }],
  },
  {
    arrayStride: 16,
    attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x4' }],
  },
];
