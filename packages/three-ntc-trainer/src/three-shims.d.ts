// See packages/three-ntc/src/three-shims.d.ts for the rationale: this `three`
// version ships no TypeScript declarations, so every import from 'three' /
// 'three/tsl' / 'three/webgpu' resolves to `any`.
declare module 'three';
declare module 'three/tsl';
declare module 'three/webgpu';
