import { float, max, normalWorld, sqrt, tangentWorld, bitangentWorld, transformNormalToView, vec2, vec3 } from 'three/tsl';

/**
 * Turns a trained tangent-space (dx, dy) offset into the mesh's final
 * view-space normal. The network only predicts the 2-component offset (see
 * NTCFormat.js's 'tanh'-activated `normal`/`clearcoatNormal`
 * channels) - z is reconstructed here as the positive root
 * `sqrt(1 - dx*dx - dy*dy)`, matching the always-positive-hemisphere
 * tangent-space z that MaterialX's own `normalNode` produces (the same
 * assumption the old full-vector encoding relied on: this z was always ~1,
 * never actually trained as an independent degree of freedom).
 *
 * This is NOT the standard three.js `normalMap(texture, scale)` convention,
 * where the returned vector is left in tangent space and the base
 * `NodeMaterial.setupNormal()` pipeline transforms it downstream using
 * whatever mesh it's applied to - `setupNormal()` actually uses `normalNode`
 * verbatim as the final normal, no further transform. MaterialX's own
 * `<normalmap>` conversion (see examples/jsm/loaders/materialx/
 * MaterialXNodeLibrary.js's mx_normalmap + MaterialXSurfaceMappings.js's
 * transformNormalToView(...) call) does the tangent/bitangent/normal blend
 * *inside* the graph itself, using whatever mesh that graph gets built
 * against - which is why baking that node's output on our flat training
 * quad and reapplying it verbatim to the torus knot was wrong for any real
 * (non-constant) bump detail: the baked value is relative to the quad's own
 * orientation, not a portable tangent-space vector, and setupNormal() never
 * re-blends it per-mesh. This function replicates that same blend here, but
 * against tangentWorld/bitangentWorld/normalWorld as resolved for whatever
 * mesh THIS material is actually applied to (the torus knot) - these are
 * dynamic per-mesh TSL accessors, so this "just works" the same way
 * MaterialX's own conversion does when applied directly to a mesh.
 *
 * Shared (via `OUTPUT_TYPES.normal.reconstruct` below) by both the `normal`
 * and `clearcoatNormal` channel descriptors in NTCFormat.js -
 * moved here, out of NTCNodeMaterial.js, so it's reusable by any
 * channel descriptor (built-in or caller-supplied) that declares itself
 * `type: 'normal'`, not just those two hardcoded call sites. Still
 * re-exported from NTCNodeMaterial.js for backward compatibility
 * with existing imports/tests.
 */
function reconstructFinalNormal( offsetNode: any ): any {

	const dx = offsetNode.x;
	const dy = offsetNode.y;
	const dz = sqrt( max( float( 1 ).sub( dx.mul( dx ) ).sub( dy.mul( dy ) ), float( 0 ) ) );
	const tangentSpace = vec3( dx, dy, dz );
	const blended = tangentWorld.mul( tangentSpace.x )
		.add( bitangentWorld.mul( tangentSpace.y ) )
		.add( normalWorld.mul( tangentSpace.z ) )
		.normalize();

	return transformNormalToView( blended );

}

/**
 * Converts a plain JS constant value (a bare number, or a [x,y]/[x,y,z]
 * array) into the equivalent TSL node - the JS-side counterpart of the
 * `activation`-decoded slice a channel produces when it's actively trained.
 * Used to give an `alwaysConstant` (or otherwise node-less) channel a debug-
 * preview node "for free", and by `NTCNodeMaterial.setDebugView`
 * to preview a constant channel through the exact same node-based color
 * pipeline as a trained one.
 */
function constantToNode( value: number | number[] ): any {

	if ( ! Array.isArray( value ) ) return float( value );

	return value.length === 2 ? vec2( ...value ) : vec3( ...value );

}

/** One entry of `OUTPUT_TYPES` - see the registry's doc comment below. */
export interface NTCOutputType {
	reconstruct: ( offsetNode: any ) => any;
	previewSize: number;
}

/**
 * Small registry of the composite "output types" this addon knows how to
 * reconstruct/preview beyond a bare scalar or fixed-size vector - keyed by
 * the channel descriptor's optional `type` field (see NTCFormat.
 * CHANNELS). A channel that doesn't need any of this (every plain scalar/
 * color channel) simply omits `type`, or sets it to one of the generic
 * 'float'/'float2'/'float3'/'color' labels, which have no entry here and are
 * handled by size alone - this registry only needs entries for types that
 * have real *behavior* attached (a preview-time size override, a shared
 * reconstruction function), not every label in the vocabulary.
 *
 * `normal`: both `normal` and `clearcoatNormal` train a 2-component
 * tangent-space (dx, dy) offset but are *consumed* (and previewed) as a
 * fully TBN-reconstructed 3-component view-space vector - see
 * `reconstructFinalNormal` above and `previewSize` below.
 */
const OUTPUT_TYPES: Record<string, NTCOutputType> = {
	normal: {
		// Shared reconstruction from a trained 2-component offset to a full
		// view-space normal - see reconstructFinalNormal's doc comment.
		reconstruct: reconstructFinalNormal,
		// The raw trained payload is 2-component, but by the time a value is
		// shown in a debug view it's always the fully reconstructed 3-
		// component vector (see NTCFormat.buildDebugViewColorNode) -
		// previewColor needs to be told to treat it as 3-wide, not 2, or it
		// hardcodes blue to 0 and silently drops z.
		previewSize: 3
	}
};

/** The subset of a channel descriptor `channelEffectiveType` needs. */
export interface EffectiveTypeChannel {
	type?: string;
	size: number;
}

/**
 * Resolves a channel descriptor's *effective* type label - its explicit
 * `type` if it set one, otherwise a generic size-based default ('float',
 * 'float2', 'float3') so every channel (built-in or caller-supplied) has
 * some type label to key off of without forcing every descriptor to state
 * the obvious.
 */
function channelEffectiveType( channel: EffectiveTypeChannel ): string {

	if ( channel.type ) return channel.type;

	return channel.size === 1 ? 'float' : `float${channel.size}`;

}

export { reconstructFinalNormal, constantToNode, OUTPUT_TYPES, channelEffectiveType };
