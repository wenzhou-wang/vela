// Minimal subset of the glTF 2.0 schema used by GLTFLoader.

export interface GLTFRoot {
  asset: { version: string };
  scene?: number;
  scenes?: { nodes?: number[]; name?: string }[];
  nodes?: GLTFNode[];
  meshes?: GLTFMesh[];
  materials?: GLTFMaterial[];
  accessors?: GLTFAccessor[];
  bufferViews?: GLTFBufferView[];
  buffers?: GLTFBuffer[];
  textures?: GLTFTextureDef[];
  images?: GLTFImage[];
  samplers?: GLTFSampler[];
  animations?: GLTFAnimation[];
  skins?: GLTFSkin[];
}

export interface GLTFSkin {
  name?: string;
  joints: number[];
  inverseBindMatrices?: number; // accessor of mat4 per joint
  skeleton?: number;
}

export interface GLTFAnimation {
  name?: string;
  channels: GLTFAnimationChannel[];
  samplers: GLTFAnimationSampler[];
}

export interface GLTFAnimationChannel {
  sampler: number;
  target: { node?: number; path: 'translation' | 'rotation' | 'scale' | 'weights' };
}

export interface GLTFAnimationSampler {
  input: number; // accessor of keyframe times
  output: number; // accessor of keyframe values
  interpolation?: 'LINEAR' | 'STEP' | 'CUBICSPLINE';
}

export interface GLTFNode {
  name?: string;
  mesh?: number;
  skin?: number;
  children?: number[];
  matrix?: number[];
  translation?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number];
}

export interface GLTFMesh {
  name?: string;
  primitives: GLTFPrimitive[];
  /** Default morph-target weights for the mesh. */
  weights?: number[];
  extras?: { targetNames?: string[] };
}

export interface GLTFPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
  mode?: number;
  /** Morph-target attribute accessors: each entry maps POSITION/NORMAL → accessor. */
  targets?: Record<string, number>[];
}

export interface GLTFMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number];
    baseColorTexture?: GLTFTextureRef;
    metallicFactor?: number;
    roughnessFactor?: number;
    metallicRoughnessTexture?: GLTFTextureRef;
  };
  normalTexture?: GLTFTextureRef & { scale?: number };
  occlusionTexture?: GLTFTextureRef & { strength?: number };
  emissiveTexture?: GLTFTextureRef;
  emissiveFactor?: [number, number, number];
  alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: {
    KHR_materials_emissive_strength?: { emissiveStrength: number };
  };
}

export interface GLTFTextureRef {
  index: number;
  texCoord?: number;
}

export interface GLTFAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  normalized?: boolean;
  count: number;
  type: 'SCALAR' | 'VEC2' | 'VEC3' | 'VEC4' | 'MAT2' | 'MAT3' | 'MAT4';
  min?: number[];
  max?: number[];
}

export interface GLTFBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}

export interface GLTFBuffer {
  uri?: string;
  byteLength: number;
}

export interface GLTFTextureDef {
  source?: number;
  sampler?: number;
}

export interface GLTFImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
}

export interface GLTFSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
}

export const COMPONENT_SIZE: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

export const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16,
};
