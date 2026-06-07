/**
 * vela — a WebGPU-first 3D rendering engine.
 */

// math
export * from './math';

// core
export { Object3D } from './core/Object3D';
export { Scene } from './core/Scene';
export { Camera } from './core/Camera';
export { PerspectiveCamera } from './core/PerspectiveCamera';
export { Mesh } from './core/Mesh';
export { BufferGeometry } from './core/BufferGeometry';
export { BufferAttribute } from './core/BufferAttribute';
export type { TypedArray } from './core/BufferAttribute';

// geometries
export { BoxGeometry } from './geometries/BoxGeometry';
export { SphereGeometry } from './geometries/SphereGeometry';
export { PlaneGeometry } from './geometries/PlaneGeometry';

// materials
export { Material } from './materials/Material';
export { StandardMaterial } from './materials/StandardMaterial';
export type { StandardMaterialParams } from './materials/StandardMaterial';

// textures
export { Texture } from './textures/Texture';
export type { TextureOptions } from './textures/Texture';

// lights
export { Light } from './lights/Light';
export { AmbientLight } from './lights/AmbientLight';
export { DirectionalLight } from './lights/DirectionalLight';
export { PointLight } from './lights/PointLight';

// renderer
export { WebGPURenderer } from './renderer/WebGPURenderer';
export type { RendererOptions } from './renderer/WebGPURenderer';

// controls
export { OrbitControls } from './controls/OrbitControls';

// loaders
export { GLTFLoader } from './loaders/GLTFLoader';
export type { GLTFResult } from './loaders/GLTFLoader';

export const VERSION = '0.1.0';
