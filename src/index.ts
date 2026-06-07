/**
 * vela — a WebGPU-first 3D rendering engine.
 */

// math
export * from './math';

// core
export { Object3D } from './core/Object3D';
export { Raycaster } from './core/Raycaster';
export type { Intersection } from './core/Raycaster';
export { BVH } from './core/BVH';
export { Scene } from './core/Scene';
export { Camera } from './core/Camera';
export { PerspectiveCamera } from './core/PerspectiveCamera';
export { OrthographicCamera } from './core/OrthographicCamera';
export { Mesh } from './core/Mesh';
export { LineSegments } from './core/LineSegments';
export { SkinnedMesh } from './core/SkinnedMesh';
export { InstancedMesh } from './core/InstancedMesh';
export { Skeleton } from './core/Skeleton';
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
export { LineBasicMaterial } from './materials/LineBasicMaterial';
export type { LineBasicMaterialParams } from './materials/LineBasicMaterial';

// helpers
export { GridHelper } from './helpers/GridHelper';
export { AxesHelper } from './helpers/AxesHelper';
export { Box3Helper } from './helpers/Box3Helper';
export { DirectionalLightHelper } from './helpers/DirectionalLightHelper';
export { PointLightHelper } from './helpers/PointLightHelper';

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
export { FlyControls } from './controls/FlyControls';

// loaders
export { GLTFLoader } from './loaders/GLTFLoader';
export type { GLTFResult } from './loaders/GLTFLoader';
export { GLTFExporter } from './loaders/GLTFExporter';
export type { GLTFExportResult } from './loaders/GLTFExporter';
export { SceneSerializer } from './loaders/SceneSerializer';

// animation
export { AnimationClip } from './animation/AnimationClip';
export { AnimationMixer } from './animation/AnimationMixer';
export { KeyframeTrack } from './animation/KeyframeTrack';
export type { InterpolationMode, TrackPath } from './animation/KeyframeTrack';

export const VERSION = '0.1.0';
