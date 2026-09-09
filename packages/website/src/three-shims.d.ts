// This `three` version ships no TypeScript declarations for its
// webgpu/addons subpaths (see packages/three-ntc/src/three-shims.d.ts for
// the full rationale) - every import from these resolves to `any`.
declare module 'three';
declare module 'three/webgpu';
declare module 'three/tsl';
declare module 'three/addons/*';

declare module '*.css?url' {
	const url: string;
	export default url;
}
