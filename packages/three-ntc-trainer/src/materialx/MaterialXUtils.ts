import { bool, element, float, vec3 } from 'three/tsl';

const BOOLEAN_OPERATOR_OPS = new Set(['&&', '||', '^^', '!', '==', '!=', '<', '>', '<=', '>=']);

/**
 * Opaque three/tsl node value (or a plain scalar such as a number/string
 * flowing through the same call sites). three ships no public TypeScript
 * declarations for its dynamically-proxied node graph (see
 * `packages/three-ntc-trainer/src/three-shims.d.ts`), so this is the
 * module's alias for "some TSL node or plain value" passed between the
 * MaterialX compile helpers.
 */
export type TSLNode = unknown;

// three/tsl nodes are dynamically proxied objects with no public type surface (see TSLNode
// above and three-shims.d.ts). This is the single, reviewable boundary the MaterialX module
// uses to read proxy-only members (nodeType, op, isOperatorNode, ...) or invoke dynamic node
// methods (notEqual, mul, add, ...) that have no static shape to type-check against.
// oxlint-disable-next-line typescript/no-explicit-any -- untyped third-party (three/tsl) boundary
type Dynamic = any;

/** Narrow escape hatch onto a TSLNode's dynamic (proxy-only) surface. See `Dynamic` above. */
export function asDynamic(node: TSLNode): Dynamic {
  return node as Dynamic;
}

function normalizeSpaceName(value: unknown, fallback = 'world'): string {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim().toLowerCase();
  if (normalized === '') return fallback;
  if (normalized === 'world') return 'world';
  if (normalized === 'object' || normalized === 'model') return 'object';
  return fallback;
}

function isBooleanNode(node: TSLNode): boolean {
  const dynamicNode = asDynamic(node);
  return (
    Boolean(dynamicNode) &&
    (dynamicNode.nodeType === 'bool' || (dynamicNode.isOperatorNode && BOOLEAN_OPERATOR_OPS.has(dynamicNode.op)))
  );
}

function toBooleanNode(node: TSLNode): TSLNode {
  if (!node) return bool(false);
  if (typeof node === 'boolean') return bool(node);
  if (typeof node === 'number') return bool(node !== 0);
  if (isBooleanNode(node)) return node;
  return asDynamic(node).notEqual(float(0));
}

function getComponentCountForType(type: string): number {
  if (type === 'vector2') return 2;
  if (type === 'vector3' || type === 'color3') return 3;
  return 4;
}

function toVec3Channels(input: TSLNode): TSLNode {
  return vec3(element(input, 0), element(input, 1), element(input, 2));
}

export { getComponentCountForType, isBooleanNode, normalizeSpaceName, toBooleanNode, toVec3Channels };
