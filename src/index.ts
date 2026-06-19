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
export type { SkyOptions, FogOptions } from './core/Scene';
export { Camera } from './core/Camera';
export { PerspectiveCamera } from './core/PerspectiveCamera';
export { OrthographicCamera } from './core/OrthographicCamera';
export { Mesh } from './core/Mesh';
export { LineSegments } from './core/LineSegments';
export { SkinnedMesh } from './core/SkinnedMesh';
export { InstancedMesh } from './core/InstancedMesh';
export { ParticleSystem } from './core/ParticleSystem';
export type { ParticleSystemOptions } from './core/ParticleSystem';
export { TrailRenderer } from './core/TrailRenderer';
export type { TrailOptions } from './core/TrailRenderer';
export { Sprite } from './core/Sprite';
export { LOD } from './core/LOD';
export { TextMesh } from './core/TextMesh';
export type { TextMeshOptions } from './core/TextMesh';
export { RenderTarget } from './core/RenderTarget';
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
export { ShaderMaterial } from './materials/ShaderMaterial';
export type { ShaderMaterialOptions, UniformValue } from './materials/ShaderMaterial';
export { ShaderPass } from './renderer/ShaderPass';
export type { ShaderPassOptions } from './renderer/ShaderPass';
export { ComputeTask, storage, uniform } from './renderer/ComputeTask';
export type { ComputeTaskOptions, ComputeBinding, StorageBinding, UniformBinding } from './renderer/ComputeTask';

// helpers
export { GridHelper } from './helpers/GridHelper';
export { AxesHelper } from './helpers/AxesHelper';
export { Box3Helper } from './helpers/Box3Helper';
export { DirectionalLightHelper } from './helpers/DirectionalLightHelper';
export { PointLightHelper } from './helpers/PointLightHelper';
export { Stats } from './helpers/Stats';

// textures
export { Texture } from './textures/Texture';
export type { TextureOptions } from './textures/Texture';
export { DataTexture, CompressedDataTexture } from './textures/DataTexture';
export { gradientTexture } from './textures/gradientTexture';
export type { GradientStop } from './textures/gradientTexture';

// lights
export { Light } from './lights/Light';
export { AmbientLight } from './lights/AmbientLight';
export { DirectionalLight } from './lights/DirectionalLight';
export { PointLight } from './lights/PointLight';
export { SpotLight } from './lights/SpotLight';

// renderer
export { WebGPURenderer } from './renderer/WebGPURenderer';
export type { RendererOptions, PixelData } from './renderer/WebGPURenderer';
export type { ToneMapping } from './renderer/PostProcessing';
export { diagnoseScene } from './renderer/diagnose';
export type { Diagnostic, DiagnoseState } from './renderer/diagnose';
export { describeScene } from './core/describe';
export type { SceneDescription } from './core/describe';
export type { RenderReport, PerfSuggestion } from './renderer/WebGPURenderer';

// golden-image testing helpers (deterministic capture + perceptual compare)
export { captureFrame, comparePixels, loadPixels, pixelsToPNG, expectFrame, FrameMismatchError } from './test';
export type { CaptureOptions, CompareOptions, CompareResult } from './test';

// controls
export { OrbitControls } from './controls/OrbitControls';
export { FlyControls } from './controls/FlyControls';

// loaders
export { GLTFLoader } from './loaders/GLTFLoader';
export type { GLTFResult } from './loaders/GLTFLoader';
export { WorkerGLTFLoader } from './loaders/WorkerGLTFLoader';
export { RGBELoader } from './loaders/RGBELoader';
export { KTX2Loader } from './loaders/KTX2Loader';
export type { BasisTranscoder, KTX2Image } from './loaders/KTX2Loader';
export type { MeshoptDecoder, KTX2TextureLoader, DracoDecoder, DracoGeometry } from './loaders/decoders';
export { GLTFExporter } from './loaders/GLTFExporter';
export type { GLTFExportResult } from './loaders/GLTFExporter';
export { SceneSerializer } from './loaders/SceneSerializer';

// animation
export { AnimationClip } from './animation/AnimationClip';
export type { AnimationEvent } from './animation/AnimationClip';
export { AnimationMixer } from './animation/AnimationMixer';
export { AnimationAction } from './animation/AnimationAction';
export { solveTwoBoneIK } from './animation/IK';
export type { TwoBoneIKOptions } from './animation/IK';
export { AnimationStateMachine } from './animation/AnimationStateMachine';
export type {
  StateMachineDef, StateDef, TransitionDef, Condition, BlendSample,
} from './animation/AnimationStateMachine';
export { KeyframeTrack } from './animation/KeyframeTrack';
export type { InterpolationMode, TrackPath } from './animation/KeyframeTrack';

export const VERSION = '0.1.0';
