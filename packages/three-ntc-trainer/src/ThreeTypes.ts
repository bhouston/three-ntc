/**
 * Named boundary types for values that come from `three` / `three/tsl` /
 * `three/webgpu`. This pinned `three` version ships no TypeScript
 * declarations (see `three-shims.d.ts`), and the closest published
 * `@types/three` release predates the Node-material/TSL API this package
 * actually uses, so every import from those modules is untyped at its
 * origin - there is no real type to recover here, only a choice between an
 * unnamed `any` at every call site or one named, documented boundary.
 *
 * These aliases are that boundary: each name documents *what* untyped value
 * flows through it (a TSL node graph, a renderer, a material, ...) so
 * `typescript/no-explicit-any` only needs a single disable, here, instead of
 * being suppressed at dozens of call sites throughout the package.
 */

// oxlint-disable typescript/no-explicit-any -- `three` ships no .d.ts for this pinned version (see three-shims.d.ts); these are the package's single, documented boundary aliases for otherwise-untyped three/TSL values.

/** A TSL shader-node graph value, e.g. a `colorNode`, a `vec4(...)` result, or anything returned by a `three/tsl` node builder / chained node method. */
export type TSLNode = any;

/** A `THREE.WebGPURenderer` (or `THREE.WebGLRenderer`) instance. */
export type ThreeRenderer = any;

/** A `THREE.Material` / `NodeMaterial` / `MeshPhysicalNodeMaterial` instance. */
export type ThreeMaterial = any;

/** A `THREE.Texture` instance. */
export type ThreeTexture = any;

/** A `THREE.RenderTarget` instance. */
export type ThreeRenderTarget = any;

/** A `THREE.Matrix3` / `THREE.Matrix4` / other `three` math object. */
export type ThreeMath = any;

/** Any other `three` runtime object (scene graph node, buffer attribute, loader result, ...) whose real type isn't available. */
export type ThreeObject = any;
