// The `three` version this package is built against (see the root repo's
// examples/jsm fork) ships no TypeScript declarations of its own, and the
// closest published `@types/three` release predates the merged Node-material
// API (`THREE.MeshPhysicalNodeMaterial` on the main entrypoint, TSL node
// chaining like `.mul()`/`.mix()`) this package actually uses - fighting a
// stale, mismatched .d.ts is worse than no types at all here. Shorthand
// ambient modules: every import from 'three' / 'three/tsl' resolves to `any`,
// same as this file's own generously-`any`-typed TSL node signatures.
declare module 'three';
declare module 'three/tsl';
declare module 'three/webgpu';
