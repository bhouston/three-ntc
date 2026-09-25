/**
 * `three/tsl` ships no public TypeScript types (see `three-shims.d.ts`) -
 * every TSL node value (the result of `float()`/`vec3()`/`.mul()`/... and
 * every node-graph value threaded through this package) is structurally
 * opaque at the type level. This is the one alias standing in for "some TSL
 * node", documented and reused everywhere instead of a bare `any`.
 */
// oxlint-disable-next-line typescript/no-explicit-any -- three/tsl ships no public node-graph types
export type TSLNode = any;

/**
 * Loose shape of the `MeshPhysicalNodeMaterial`-like objects channel
 * descriptors (see `NTCFormat.ts`'s `CHANNELS`) read from / write onto -
 * every channel accesses a different, dynamically keyed subset of node/
 * plain properties (see `CHANNELS`' doc comment), so this is intentionally
 * just an index signature rather than a fixed property list. `three` itself
 * ships no public types either (see `three-shims.d.ts`), so this is as
 * precise as this dynamic access pattern can be typed.
 */
export type NTCMaterialLike = Record<string, unknown>;

/**
 * A resolved channel value - a plain number, or a `[x,y]`/`[x,y,z]` tuple -
 * see `NTCFormat.ts`'s `CHANNELS` (`defaultValue`/`resolveConstant`/
 * `applyConstant`/`decodeConstant`).
 */
export type NTCConstantValue = number | number[];

/**
 * Structural (`THREE.Matrix3`-shaped) view of the per-material UV transform
 * - `three` ships no public types (see `three-shims.d.ts`), so this only
 * names the `elements` access `NTCFormat.ts`'s `decodeUvTransform`/
 * `encodeUvTransform`/`isIdentityUvTransform` perform.
 */
export interface NTCMatrix3Like {
  elements: number[];
}

/**
 * Structural (`THREE.Color`-shaped) view of a resolved color property -
 * `three` ships no public types (see `three-shims.d.ts`), so this only
 * names the `r`/`g`/`b` access `NTCFormat.ts`'s color-channel descriptors
 * perform.
 */
export interface NTCColorLike {
  r: number;
  g: number;
  b: number;
}
