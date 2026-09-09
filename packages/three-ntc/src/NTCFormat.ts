import * as THREE from 'three';
import { float, transformNormalToView, vec2, vec3, vec4 } from 'three/tsl';
import { OUTPUT_TYPES, channelEffectiveType, constantToNode, reconstructFinalNormal } from './NTCOutputTypes.js';
import type { NTCActivation } from './NTCOutputActivations.js';

/**
 * The `.ntc` file format identifier/version - see `training/NTCManifest.js`
 * (encoder) and `../loaders/NTCLoader.js` (decoder). Declared here, not in
 * NTCManifest.js, since NTCLoader.js (runtime, no training dependencies)
 * needs it too, and this file is the one shared foundation both the
 * runtime decode path and the training/export path already build on.
 */
const FORMAT = 'three-ntc';
// Bumped to 2 for the mip-pyramid-of-feature-grids redesign (see
// NTCGridPyramidModel.js/NTCMipBands.js): `.ntc` files now store `levels`,
// `mipsPerLevel`, and `maxLod` as required `latents` fields, the grid
// resolution ordering is finest-first, and the decoder input is fixed at
// `channels + 1` rather than scaling with level count - a genuinely
// incompatible change from version 1 files, which this loader deliberately
// does not attempt to read (see NTCLoader.js's `validateManifest`).
const VERSION = 2;

/**
 * One entry of the material's channel vocabulary - see `CHANNELS`' doc
 * comment below for what each field means. `resolveNode`/`resolveConstant`
 * pull this channel's raw value off a source material; `applyActive`/
 * `applyConstant` apply a trained slice, or a resolved constant, onto a
 * reconstructed `NTCNodeMaterial`. Node types are left as `any` - they are
 * TSL node-graph values, not plain JS values.
 */
export interface NTCChannel {
	key: string;
	size: number;
	activation?: NTCActivation;
	type?: string;
	nodeKeys: string[];
	clampRange: [number, number] | null;
	defaultValue: any;
	resolveNode: ( material: any ) => any;
	resolveConstant: ( material: any ) => any;
	applyActive: ( targetMaterial: any, sliceNode: any ) => void;
	applyConstant: ( targetMaterial: any, constantValue: any ) => void;
	decodeConstant?: ( value: any ) => any;
	transformBakeComponent?: ( component: any, index: number ) => any;
}

/** A channel descriptor assigned a flat offset by `layoutChannels`. */
export interface NTCLayoutChannel extends NTCChannel {
	offset: number;
}

/**
 * Full vocabulary of standard PBR channels this trainer knows how to fit,
 * jointly, with one shared grid + MLP network (NVIDIA neural texture
 * compression style: one small decoder, many correlated output channels).
 *
 * A given material only *trains* the subset of these channels it actually
 * drives with a node graph (texture/procedural) - see
 * `NTCSource.classifyMaterialChannels` - so the network's output
 * width, and the sample cost of every training iteration, scale with how
 * much of the material is actually spatially-varying rather than always
 * paying for all of them. `nodeKeys` is what that classification checks:
 * a channel is "active" (worth training) iff at least one of its node
 * properties is set on the source material; otherwise it's read once as a
 * plain constant and applied directly, bypassing the network entirely.
 *
 * `activation` is the output nonlinearity applied to this channel's slice of
 * the decoder's (always-linear, see NTCGridPyramidModel.js) raw output - see
 * ./NTCOutputActivations.js - chosen to match each channel's natural
 * physical range instead of forcing every channel through the same
 * unbounded linear output:
 *  - 'sigmoid': bounded [0,1] scalars/colors (reflectance, coverage,
 *    roughness-like properties).
 *  - 'tanh': bounded [-1,1] signed vectors (tangent-space normal offsets,
 *    anisotropy direction) - trained/consumed directly in that range, no
 *    extra `*0.5+0.5` remap needed.
 *  - 'softplus': unbounded, non-negative HDR values (emission) - smooth and
 *    non-exploding, unlike a raw exponential, for an undertrained decoder's
 *    large pre-activations.
 * Anything without an explicit `activation` falls back to plain linear
 * (matching every other MLP output in this codebase).
 */

/**
 * `clampRange` and `defaultValue` (alongside `activation`/`nodeKeys`/`size`
 * above) are what let `NTCNodeMaterial`'s constructor and
 * `NTCSource`'s node/constant resolvers be driven by a loop
 * instead of one hand-written branch per channel.
 *
 * `clampRange` is the range this channel's *trained* value is clamped to
 * before use (a trained value has no hard guarantee of landing in its
 * activation's nominal range - see `previewColor`'s doc comment). `null`
 * means "don't clamp" - used for vector-valued channels that get a
 * different, channel-specific treatment instead (normal reconstruction,
 * anisotropy's hypot/atan2, unbounded emission).
 *
 * `defaultValue` is what a *constant* (untrained) channel falls back to when
 * the source material doesn't set even the corresponding plain property -
 * matching three.js's own `MeshPhysicalMaterial` defaults for that property.
 *
 * A constant channel's resolved value is applied via `applyConstant` as a
 * literal TSL constant node (`float(...)`/`vec2(...)`/`vec3(...)`) assigned
 * onto the reconstructed material's `*Node` property - not the plain
 * (non-`Node`) property. Setting the plain property (e.g. `material.roughness
 * = 0.4`) would route through three.js's `MaterialReferenceNode` machinery
 * (see `materialRoughness` in src/nodes/accessors/MaterialNode.js), which
 * reads `material.roughness` back out as a live GPU uniform every frame - a
 * value the shader compiler can never constant-fold, dead-code-eliminate
 * around, or otherwise specialize on, because from its perspective it's
 * indistinguishable from a value that might change later. Assigning a TSL
 * constant node instead bakes the literal directly into the generated
 * shader source, exactly like an *active* (trained) channel's slice does -
 * so a constant channel and a trained one differ only in whether their value
 * came from the network or from the source material, not in how "constant"
 * either one is to the compiler.
 *
 * This is an *open, composable* vocabulary, not a hardcoded switch: every
 * channel below carries its own `resolveNode`/`resolveConstant` (how to pull
 * this channel's raw value off a source `MeshPhysicalNodeMaterial`, as a TSL
 * node or a plain JS value respectively - see NTCSource.js) and
 * `applyActive`/`applyConstant` (how to apply a trained slice, or a resolved
 * constant, onto a reconstructed `NTCNodeMaterial` - see
 * NTCNodeMaterial.js). `NTCSource.js` and
 * `NTCNodeMaterial.js` are both just generic loops driven by
 * these four functions per channel - there's no per-key branching left in
 * either file, so a caller can compose an entirely different channel array
 * (new named outputs, with their own resolution/application logic) and pass
 * it through `classifyMaterialChannels`/`resolveMaterialChannelNodes`/
 * `NTCNodeMaterial`'s constructor via their `channels` option,
 * without editing this file at all. `CHANNELS` below is simply the built-in,
 * default vocabulary those entry points fall back to.
 *
 * `type` is an optional label (see NTCOutputTypes.js's `OUTPUT_TYPES`/
 * `channelEffectiveType`) identifying a channel's reconstruction/preview
 * shape beyond its raw trained `size` - e.g. `'normal'` for a channel that
 * trains a 2-component tangent-space offset but reconstructs/previews as a
 * full 3-component vector. Channels that don't need this can omit it; it
 * defaults to a generic size-based label ('float'/'float2'/'float3').
 */

function resolveScalarNode( material: any, nodeKey: string, propertyKey: string, fallback: number ): any {

	if ( material[ nodeKey ] ) return float( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return float( material[ propertyKey ] );

	return float( fallback );

}

function resolveColorNode( material: any, nodeKey: string, propertyKey: string, fallback: any ): any {

	if ( material[ nodeKey ] ) return vec3( material[ nodeKey ] );
	if ( material[ propertyKey ] !== undefined ) return vec3( material[ propertyKey ] );

	return vec3( fallback );

}

/**
 * Builds a regular "single `${key}Node`/`${key}` node/property pair, scalar
 * clamp range, scalar default" channel descriptor - the shape most PBR
 * channels (roughness, metalness, transmission, ...) actually have. Channels
 * that don't fit this shape (a Color + separate intensity property, a
 * reconstructed vector, a decomposed strength/rotation pair) are written out
 * by hand below instead of forcing them through this factory.
 */
function simpleScalarChannel( key: string, { activation, clampRange, defaultValue }: { activation?: NTCActivation; clampRange: [number, number] | null; defaultValue: number } ): NTCChannel {

	return {
		key,
		size: 1,
		activation,
		nodeKeys: [ key + 'Node' ],
		clampRange,
		defaultValue,
		resolveNode: ( material: any ) => resolveScalarNode( material, key + 'Node', key, defaultValue ),
		resolveConstant: ( material: any ) => ( material[ key ] !== undefined ? material[ key ] : defaultValue ),
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial[ key + 'Node' ] = clampRange ? sliceNode.clamp( ...clampRange ) : sliceNode;

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			// A literal `float(...)` constant node, not the plain `key`
			// property - see this file's top doc comment for why.
			targetMaterial[ key + 'Node' ] = float( constantValue );

		}
	};

}

/**
 * Builds a "Color property + separate scalar intensity property" channel
 * descriptor - the shape `emissive`/`sheenColor` have (`emissive` +
 * `emissiveIntensity`, `sheenColor` + `sheen`). The trained/resolved node and
 * constant value are always the color already multiplied by its intensity -
 * matching how the source material's own shading consumes them - so this
 * channel doesn't separately reconstruct an intensity on the way back out;
 * `applyConstant` (passed in by each caller below) writes only the `*Node`
 * property, as a literal constant node with the intensity already folded in,
 * matching this addon's existing behavior of not round-tripping a distinct
 * intensity scalar for these two channels.
 */
function colorIntensityChannel( key: string, { nodeKey, colorProperty, intensityProperty, applyActive, applyConstant }: {
	nodeKey: string;
	colorProperty: string;
	intensityProperty: string;
	applyActive: ( targetMaterial: any, sliceNode: any ) => void;
	applyConstant: ( targetMaterial: any, constantValue: any ) => void;
} ): NTCChannel {

	return {
		key,
		size: 3,
		activation: 'sigmoid',
		type: 'color',
		nodeKeys: [ nodeKey ],
		clampRange: [ 0, 1 ],
		defaultValue: [ 0, 0, 0 ],
		resolveNode: ( material: any ) => {

			// `material[nodeKey]`, when set, is *already* the fully-resolved
			// color*intensity value - exactly what MeshPhysicalNodeMaterial's
			// own shading assigns it as-is, with no further `*intensityProperty`
			// scaling (see `emissive.assign(vec3(emissiveNode ? emissiveNode :
			// materialEmissive))` / the equivalent `sheenNode` line in
			// src/materials/nodes/MeshPhysicalNodeMaterial.js - the intensity
			// multiply only happens *inside* `materialEmissive`/`materialSheen`'s
			// own plain-property fallback, in src/nodes/accessors/MaterialNode.js).
			// Multiplying it by `intensityProperty` again here - as this used to
			// do unconditionally - silently zeroed out every MaterialX-sourced
			// sheen material: `MaterialXSurfaceMappings.js` always sets
			// `sheenNode` directly (already `sheen_weight * sheen_color`) and
			// never touches the separate `material.sheen` scalar, which defaults
			// to `0` (unlike `emissiveIntensity`'s default of `1`, which
			// happened to make the same bug a no-op for `emissive` - by luck,
			// not by design). Only the plain-property fallback path below still
			// needs the explicit multiply, since three.js itself only folds
			// color*intensity together when resolving *that* path.
			if ( material[ nodeKey ] ) return vec3( material[ nodeKey ] );

			const color = material[ colorProperty ] || new THREE.Color( 0, 0, 0 );
			const intensity = material[ intensityProperty ] !== undefined ? material[ intensityProperty ] : 1;
			return vec3( color ).mul( intensity );

		},
		resolveConstant: ( material: any ) => {

			const c = material[ colorProperty ] || new THREE.Color( 0, 0, 0 );
			const i = material[ intensityProperty ] !== undefined ? material[ intensityProperty ] : 1;
			return [ c.r * i, c.g * i, c.b * i ];

		},
		applyActive,
		applyConstant
	};

}

/**
 * Builds a scalar channel descriptor for a physical quantity that has a
 * fixed, known-in-advance range (e.g. an index of refraction) but isn't
 * itself naturally [0,1]-bounded. Unlike `simpleScalarChannel`'s bare
 * `clampRange` (a post-hoc clamp applied *after* an otherwise-unconstrained
 * activation), this trains strictly as a [0,1] fraction of `[min, max]` via
 * 'sigmoid' - the network is architecturally incapable of ever representing
 * a value outside that range, not just clamped into it after the fact - and
 * decodes back to physical units (`min + fraction * (max - min)`) on the way
 * out. `min`/`max` are a fixed property of the channel itself, not derived
 * per-material (contrast `iridescenceThickness` below, whose range *is*
 * per-material - there's no equivalent per-material "IOR range" concept in
 * three.js/glTF to read one from).
 */
function fixedRangeScalarChannel( key: string, { min, max, defaultValue }: { min: number; max: number; defaultValue: number } ): NTCChannel {

	const range = max - min;

	return {
		key,
		size: 1,
		activation: 'sigmoid',
		nodeKeys: [ key + 'Node' ],
		clampRange: null,
		defaultValue,
		resolveNode: ( material: any ) => {

			const raw = resolveScalarNode( material, key + 'Node', key, defaultValue );
			return raw.sub( min ).div( range ).clamp( 0, 1 );

		},
		resolveConstant: ( material: any ) => ( material[ key ] !== undefined ? material[ key ] : defaultValue ),
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial[ key + 'Node' ] = sliceNode.clamp( 0, 1 ).mul( range ).add( min );

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			// The physical value itself (not a fraction) - matches every other
			// channel's `applyConstant`/`resolveConstant` pairing; only the
			// *trained* path needs the [0,1] fraction round trip above.
			targetMaterial[ key + 'Node' ] = float( constantValue );

		}
	};

}

/**
 * Builds a "plain Color property, no separate intensity" channel descriptor -
 * the shape `attenuationColor` has (unlike `emissive`/`sheenColor`, which pair
 * a Color with a separate intensity scalar - see `colorIntensityChannel`
 * above). Trained/resolved directly as the Color's (r, g, b), sigmoid-bounded
 * to [0,1] like every other reflectance-style color channel.
 */
function simpleColorChannel( key: string, { defaultValue }: { defaultValue: [ number, number, number ] } ): NTCChannel {

	const fallbackColor = () => new THREE.Color( ...defaultValue );

	return {
		key,
		size: 3,
		activation: 'sigmoid',
		type: 'color',
		nodeKeys: [ key + 'Node' ],
		clampRange: [ 0, 1 ],
		defaultValue,
		resolveNode: ( material: any ) => resolveColorNode( material, key + 'Node', key, fallbackColor() ),
		resolveConstant: ( material: any ) => {

			const c = material[ key ] || fallbackColor();
			return [ c.r, c.g, c.b ];

		},
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial[ key + 'Node' ] = sliceNode.clamp( 0, 1 );

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial[ key + 'Node' ] = vec3( ...constantValue );

		}
	};

}

/**
 * Builds one endpoint (`index` 0 or 1) of a per-material *metadata* channel
 * for `material.iridescenceThicknessRange` - read-only, never node-driven,
 * and (deliberately) never trainable: `nodeKeys` names a property that can
 * never exist on a real material, so `classifyMaterialChannels`'s `hasNode`
 * check is always false and this channel always takes the (per-material,
 * live) `resolveConstant` path, regardless of whether `iridescenceThickness`
 * itself is active. This is what "output the min/max as constants from the
 * original material" actually means here: the range endpoints are read
 * straight off the source material and carried through `constantValues`
 * as-is - never fit, never adapted, never treated as something the network
 * could output. See the `iridescenceThickness` channel below for how these
 * two are consumed on the way back in (`applyConstant`/`applyActive` write
 * `targetMaterial.iridescenceThicknessRange[index]`/read it back,
 * respectively - both rely on this channel already having run first, so it
 * must stay ordered before `iridescenceThickness` in `CHANNELS`).
 */
function iridescenceThicknessRangeChannel( index: 0 | 1, fallback: number ): NTCChannel {

	const key = index === 0 ? 'iridescenceThicknessRangeMin' : 'iridescenceThicknessRangeMax';

	return {
		key,
		size: 1,
		nodeKeys: [ '__never__' ],
		clampRange: null,
		defaultValue: fallback,
		resolveNode: ( material: any ) => float( material.iridescenceThicknessRange ? material.iridescenceThicknessRange[ index ] : fallback ),
		resolveConstant: ( material: any ) => ( material.iridescenceThicknessRange ? material.iridescenceThicknessRange[ index ] : fallback ),
		applyActive: () => {}, // never reached - see nodeKeys above
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial.iridescenceThicknessRange[ index ] = constantValue;

		}
	};

}

/**
 * Builds a `type: 'normal'` channel descriptor - a 2-component tangent-space
 * (dx, dy) offset trained/packed by the network, but reconstructed into a
 * full 3-component view-space vector at consumption time - see
 * NTCOutputTypes.js's `reconstructFinalNormal`/`OUTPUT_TYPES.normal`.
 * Shared by `normal` and `clearcoatNormal` below, which differ only in which
 * node/property pair they read from and write to.
 */
function normalChannel( key: string, materialNodeProperty: string ): NTCChannel {

	return {
		key,
		size: 2,
		activation: 'tanh',
		type: 'normal',
		nodeKeys: [ materialNodeProperty ],
		clampRange: null,
		defaultValue: [ 0, 0, 1 ],
		resolveNode: ( material: any ) => ( material[ materialNodeProperty ] ? vec3( material[ materialNodeProperty ] ) : vec3( 0, 0, 1 ) ),
		resolveConstant: () => [ 0, 0, 1 ],
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial[ materialNodeProperty ] = OUTPUT_TYPES.normal.reconstruct( sliceNode );

		},
		// A constant normal/clearcoatNormal channel means "no bump" - leave
		// the corresponding *Node property unset entirely, rather than wiring
		// up a node that always evaluates to the flat [0,0,1] default.
		applyConstant: () => {}
	};

}

/**
 * Finite stand-in for `Infinity` - see the `attenuationDistance` channel
 * below for the full reasoning. Exactly representable in a 32-bit float
 * (well under 2^24), and far outside the range of any physically plausible
 * scene-unit attenuation distance, so it's unambiguous to recognize once
 * decoded (see `decodeConstantValues`).
 */
const ATTENUATION_DISTANCE_INFINITY_SENTINEL = 1e6;

const CHANNELS: NTCChannel[] = [
	{
		key: 'albedo', size: 3, activation: 'sigmoid', type: 'color', nodeKeys: [ 'colorNode' ], clampRange: null, defaultValue: [ 1, 1, 1 ],
		resolveNode: ( material: any ) => ( material.colorNode ? vec3( material.colorNode ) : vec3( material.color || new THREE.Color( 1, 1, 1 ) ) ),
		resolveConstant: ( material: any ) => {

			const c = material.color || new THREE.Color( 1, 1, 1 );
			return [ c.r, c.g, c.b ];

		},
		// Also stashed as `_shadedColorNode` - swapped back in by
		// NTCNodeMaterial.setDebugView('shaded') - since albedo is
		// the one channel whose trained slice doubles as the material's real,
		// lit `colorNode`.
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial._shadedColorNode = sliceNode;
			targetMaterial.colorNode = sliceNode;

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial._shadedColorNode = vec3( ...constantValue );
			targetMaterial.colorNode = targetMaterial._shadedColorNode;

		}
	},
	simpleScalarChannel( 'opacity', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 1 } ),
	// 2-component tangent-space (dx, dy) offset, not a full xyz vector - z is
	// reconstructed at consumption time as sqrt(1 - dx*dx - dy*dy), see
	// NTCOutputTypes.reconstructFinalNormal.
	normalChannel( 'normal', 'normalNode' ),
	simpleScalarChannel( 'roughness', { activation: 'sigmoid', clampRange: [ 0.02, 1 ], defaultValue: 1 } ),
	simpleScalarChannel( 'metalness', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } ),
	simpleScalarChannel( 'clearcoat', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } ),
	simpleScalarChannel( 'clearcoatRoughness', { activation: 'sigmoid', clampRange: [ 0.02, 1 ], defaultValue: 0 } ),
	normalChannel( 'clearcoatNormal', 'clearcoatNormalNode' ),
	// Thin-film interference (glTF KHR_materials_iridescence /
	// MeshPhysicalMaterial.iridescence*) - a distinct surface layer from
	// clearcoat, spatially-varying on real-world sources (oil slicks,
	// anodized metal, soap film) via its strength and film thickness alike,
	// so both get their own trainable channel rather than being folded into
	// a single "iridescence" scalar.
	simpleScalarChannel( 'iridescence', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } ),
	// Index of refraction of the thin film itself - almost always spatially
	// *constant* in practice (it's a material constant of the film, not
	// something that's painted on), but trainable all the same for
	// consistency with every other channel here. Trained strictly as a [0,1]
	// fraction of a fixed [1.0, 2.333] window (glTF's own spec range) via
	// `fixedRangeScalarChannel`, not an unbounded value merely clamped after
	// the fact - the network is architecturally incapable of ever
	// representing an out-of-range IOR, and stays in the same well-behaved
	// (saturates both ends) 'sigmoid' regime every other bounded channel
	// here trains in.
	fixedRangeScalarChannel( 'iridescenceIOR', { min: 1.0, max: 2.333, defaultValue: 1.3 } ),
	// Thin-film thickness in nanometers. Unlike `iridescenceIOR` above, this
	// doesn't train against a *fixed* range - it trains as a [0,1] fraction
	// of the *source material's own* `iridescenceThicknessRange` (exactly
	// the convention three.js/glTF's own `iridescenceThicknessMap` already
	// uses: its green channel is a 0-1 fraction remapped by that same range
	// - see MaterialNode.js's IRIDESCENCE_THICKNESS resolution), so the
	// network never has to learn a range it wasn't told about, and never
	// sees (or could output) a raw nanometer value outside [0,1]. The two
	// `iridescenceThicknessRangeChannel` entries just below carry the range
	// endpoints themselves through `constantValues`, verbatim off the
	// source material - never trained, never adapted - and must run first
	// in this array so `targetMaterial.iridescenceThicknessRange` is
	// already correct by the time this channel's `applyActive`/
	// `applyConstant` read it back.
	iridescenceThicknessRangeChannel( 0, 100 ),
	iridescenceThicknessRangeChannel( 1, 400 ),
	{
		key: 'iridescenceThickness', size: 1, activation: 'sigmoid', nodeKeys: [ 'iridescenceThicknessNode' ], clampRange: [ 0, 1 ], defaultValue: 1,
		resolveNode: ( material: any ) => {

			// No map at all -> always the declared maximum (fraction 1),
			// matching MaterialNode.js's own IRIDESCENCE_THICKNESS fallback.
			if ( ! material.iridescenceThicknessNode ) return float( 1 );

			const [ min, max ] = material.iridescenceThicknessRange || [ 100, 400 ];
			return float( material.iridescenceThicknessNode ).sub( min ).div( Math.max( max - min, 1e-6 ) ).clamp( 0, 1 );

		},
		resolveConstant: () => 1, // same "no map -> maximum" fallback as above
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			const [ min, max ] = targetMaterial.iridescenceThicknessRange;
			targetMaterial.iridescenceThicknessNode = float( min ).add( sliceNode.clamp( 0, 1 ).mul( max - min ) );

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			const [ min, max ] = targetMaterial.iridescenceThicknessRange;
			targetMaterial.iridescenceThicknessNode = float( min + constantValue * ( max - min ) );

		}
	},
	simpleScalarChannel( 'transmission', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } ),
	// Dielectric specular reflectance multiplier - MeshPhysicalMaterial
	// defaults this to 1 (full Fresnel reflectance at normal incidence, per
	// `ior`), but a source material can turn its non-metal specular lobe down
	// or fully off (e.g. a MaterialX `specular="0"` Lambertian). Without this
	// channel that override was silently dropped on reconstruction - the
	// neural material always inherited MeshPhysicalMaterial's `specularIntensity
	// = 1` default regardless of what the source material set - producing a
	// reconstruction that looks visibly shinier than a matte source despite
	// matching roughness/metalness.
	simpleScalarChannel( 'specularIntensity', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 1 } ),
	// Tint of the dielectric specular lobe `specularIntensity` scales - a
	// separate channel from `specularIntensity` for the same reason
	// `sheenColor`/`sheen` are separate: one is a color, one is a scalar
	// multiplier, and a source material can vary either independently.
	simpleColorChannel( 'specularColor', { defaultValue: [ 1, 1, 1 ] } ),
	// Base dielectric index of refraction - drives both the Fresnel specular
	// response above and, when `transmission` is active, the refraction
	// through the volume. Same `fixedRangeScalarChannel` treatment (and the
	// same glTF-spec-derived bound) as `iridescenceIOR`.
	fixedRangeScalarChannel( 'ior', { min: 1.0, max: 2.333, defaultValue: 1.5 } ),
	// Transmission *volume* properties - how far light travels, and how it's
	// tinted, once it's inside the dielectric behind a `transmission`
	// surface, as opposed to `transmission` itself (the fraction of light
	// that enters the surface at all). `thickness` is a local scene-unit
	// distance (unbounded, non-negative - 'softplus', same reasoning as
	// `iridescenceIOR`/`iridescenceThickness` above) rather than a [0,1]
	// fraction, so it deliberately isn't `clampRange`-bounded the way every
	// sigmoid-activated channel here is.
	simpleScalarChannel( 'thickness', { activation: 'softplus', clampRange: null, defaultValue: 0 } ),
	simpleColorChannel( 'attenuationColor', { defaultValue: [ 1, 1, 1 ] } ),
	// Absorption falloff distance - `MeshPhysicalMaterial.attenuationDistance`
	// defaults to `Infinity` ("no absorption"/color-preserving as light
	// travels through the volume), which is both a perfectly meaningful value
	// to *apply* (a literal `float(Infinity)` TSL constant is not valid
	// shader source, but a sufficiently large finite one is visually and
	// numerically indistinguishable - Beer-Lambert extinction over any
	// on-screen distance is already ~1 well before this) and an awkward one
	// to *persist* (`JSON.stringify(Infinity)` silently degrades to `null`,
	// which would then fail to round-trip back into a usable constant at
	// all). `ATTENUATION_DISTANCE_INFINITY_SENTINEL` below is that finite
	// stand-in - only ever relevant on the *constant* (untrained) path, since
	// a trained slice always decodes to some finite softplus output; see
	// this channel's `decodeConstant` (and `decodeConstantValues`'s doc
	// comment) for the other half of this - converting the sentinel back to
	// a real `Infinity` once it's safely out of JSON, for any caller that
	// wants the true semantic value rather than the encoded one.
	( () => {

		const key = 'attenuationDistance';
		const nodeKey = key + 'Node';

		const encode = ( value: number ) => ( Number.isFinite( value ) ? value : ATTENUATION_DISTANCE_INFINITY_SENTINEL );

		return {
			key, size: 1, activation: 'softplus' as NTCActivation, nodeKeys: [ nodeKey ],
			clampRange: [ 0, ATTENUATION_DISTANCE_INFINITY_SENTINEL ] as [ number, number ], defaultValue: ATTENUATION_DISTANCE_INFINITY_SENTINEL,
			resolveNode: ( material: any ) => {

				if ( material[ nodeKey ] ) return float( material[ nodeKey ] );
				return float( encode( material.attenuationDistance !== undefined ? material.attenuationDistance : Infinity ) );

			},
			resolveConstant: ( material: any ) => encode( material.attenuationDistance !== undefined ? material.attenuationDistance : Infinity ),
			applyActive: ( targetMaterial: any, sliceNode: any ) => {

				targetMaterial[ nodeKey ] = sliceNode.clamp( 0, ATTENUATION_DISTANCE_INFINITY_SENTINEL );

			},
			applyConstant: ( targetMaterial: any, constantValue: any ) => {

				targetMaterial[ nodeKey ] = float( constantValue );

			},
			// See this channel's doc comment above - only meaningful for a
			// *constant* value (a trained one is never the sentinel, since
			// softplus never produces it exactly).
			decodeConstant: ( value: number ) => ( value === ATTENUATION_DISTANCE_INFINITY_SENTINEL ? Infinity : value )
		};

	} )(),
	colorIntensityChannel( 'emissive', {
		nodeKey: 'emissiveNode', colorProperty: 'emissive', intensityProperty: 'emissiveIntensity',
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial.emissiveNode = sliceNode;

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial.emissiveNode = vec3( ...constantValue );

		}
	} ),
	// Anisotropy strength+rotation trained as a single signed 2D direction
	// vector (ax, ay) = (cos(rotation)*strength, sin(rotation)*strength),
	// exactly what MeshPhysicalMaterial.anisotropyNode itself consumes. This
	// avoids training a wrapping [0,1] angle (which has a hard 0/1
	// discontinuity and an atan2(0,0) singularity at zero strength) through a
	// bounded activation.
	{
		key: 'anisotropy', size: 2, activation: 'tanh', type: 'anisotropyVector', nodeKeys: [ 'anisotropyNode' ], clampRange: null, defaultValue: [ 0, 0 ],
		resolveNode: ( material: any ) => {

			if ( material.anisotropyNode ) return vec2( material.anisotropyNode );

			const strength = material.anisotropy !== undefined ? material.anisotropy : 0;
			const rotation = material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;

			return vec2( Math.cos( rotation ) * strength, Math.sin( rotation ) * strength );

		},
		resolveConstant: ( material: any ) => {

			const strength = material.anisotropy !== undefined ? material.anisotropy : 0;
			const rotation = material.anisotropyRotation !== undefined ? material.anisotropyRotation : 0;
			return [ Math.cos( rotation ) * strength, Math.sin( rotation ) * strength ];

		},
		// Trained directly as a signed (ax, ay) direction vector, exactly the
		// form MeshPhysicalMaterial.anisotropyNode itself expects - no cos/sin
		// recombination needed.
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial.anisotropyNode = sliceNode;

		},
		// Applied as the same raw (ax, ay) direction node `anisotropyNode`
		// itself expects - no hypot/atan2 decomposition into the legacy
		// strength/rotation properties needed, since those would (like every
		// other plain property) route through a live uniform instead of a
		// literal constant node. The generic "skip applyConstant entirely
		// when the resolved value equals this channel's own `defaultValue`"
		// pass below (see its doc comment) is what actually keeps a
		// zero-strength material from ever reaching this assignment - this
		// closure doesn't need to know that itself.
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial.anisotropyNode = vec2( ...constantValue );

		}
	},
	colorIntensityChannel( 'sheenColor', {
		nodeKey: 'sheenNode', colorProperty: 'sheenColor', intensityProperty: 'sheen',
		applyActive: ( targetMaterial: any, sliceNode: any ) => {

			targetMaterial.sheenNode = sliceNode.clamp( 0, 1 );

		},
		applyConstant: ( targetMaterial: any, constantValue: any ) => {

			targetMaterial.sheenNode = vec3( ...constantValue );

		}
	} ),
	simpleScalarChannel( 'sheenRoughness', { activation: 'sigmoid', clampRange: [ 0.02, 1 ], defaultValue: 1 } ),
	// Chromatic dispersion strength (KHR_materials_dispersion) - conventional
	// content stays within [0,1] (glTF: "typically in the range of 0 to 1,
	// but higher values are unrestricted"), so this is 'sigmoid'-activated,
	// like every other naturally-bounded reflectance-style scalar here,
	// rather than 'softplus' - an out-of-range source value beyond 1 would
	// still just clamp to 1 like `roughness`/`clearcoat` do at their own
	// bounds.
	simpleScalarChannel( 'dispersion', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } ),
	simpleScalarChannel( 'retroreflectivity', { activation: 'sigmoid', clampRange: [ 0, 1 ], defaultValue: 0 } )
];

/**
 * The normal/clearcoatNormal channels' baked training target needs one extra
 * per-component transform their y-component doesn't share with every other
 * channel - see `flattenChannelComponents` in NTCSource.js. Left
 * as an explicit, opt-in hook (`transformBakeComponent`) on just those two
 * descriptors, rather than a `type`-keyed lookup, since it's tied to how the
 * training quad itself is baked (a UV/tangent convention artifact of
 * `bakeColorNodeToTexture`), not to the `'normal'` output type's
 * reconstruction semantics - a custom `type: 'normal'` channel that isn't
 * baked the same way wouldn't want this by default.
 */
for ( const channel of CHANNELS ) {

	if ( channel.type !== 'normal' ) continue;

	// The normal/clearcoatNormal offset is baked on a flat quad whose UV.v
	// gets flipped before computeTangents() runs (see bakeColorNodeToTexture
	// in NTCTextureSource.js - that flip is required for correct
	// scalar-channel UV round-tripping), which flips that quad's bitangent
	// handedness relative to any real mesh's own (un-flipped) tangent basis.
	// Since the quad sits at the identity transform, its tangent/bitangent/
	// normal are otherwise meant to coincide with world X/Y/Z, so the baked y
	// here is the *negated* tangent-space offset the network is meant to
	// learn. Negate it back so the trained (dx, dy) matches the true
	// tangent-space convention that OUTPUT_TYPES.normal.reconstruct blends
	// through a real mesh's un-flipped tangentWorld/bitangentWorld at
	// inference time - without this, the neural material's reconstructed
	// normal disagrees in sign with the teacher's live-evaluated one (visible
	// as a mismatched "normal" debug view between the two).
	channel.transformBakeComponent = ( component: any, index: number ) => ( index === 1 ? component.negate() : component );

}

/**
 * True iff `value` (a plain number, or a [x,y]/[x,y,z] array - the shape
 * every channel's `resolveConstant`/`constantValue` produces) is exactly
 * this channel's own `defaultValue` - used by the pass below to skip
 * `applyConstant` entirely for a channel whose resolved value is a total
 * no-op, rather than special-casing that per channel.
 */
function constantEqualsDefault( channel: NTCChannel, value: any ): boolean {

	const defaultValue = channel.defaultValue;
	if ( defaultValue === undefined ) return false;

	if ( Array.isArray( defaultValue ) ) {

		return Array.isArray( value ) && value.length === defaultValue.length && value.every( ( v: any, i: number ) => v === defaultValue[ i ] );

	}

	return value === defaultValue;

}

/**
 * Wraps every built-in channel's `applyConstant` so it's skipped entirely
 * when the resolved constant value is exactly this channel's own
 * `defaultValue` - i.e. a value that changes nothing about how the material
 * should shade, matching what a freshly-constructed `MeshPhysicalNodeMaterial`
 * already does by default. This isn't just an optimization: several PBR
 * properties gate an entire extra shading branch on whether their `*Node`
 * property is non-null at all - `MeshPhysicalNodeMaterial`'s `useClearcoat`/
 * `useSheen`/`useAnisotropy`/`useTransmission` getters (src/materials/nodes/
 * MeshPhysicalNodeMaterial.js) all check `this.<key>Node !== null`,
 * independent of what value that node actually evaluates to. Assigning a
 * literal *default-valued* constant node (e.g. `anisotropyNode = vec2(0, 0)`)
 * still satisfies that check, so it force-enables a branch - including, for
 * anisotropy, a `TBNViewMatrix` tangent/bitangent frame built from
 * screen-space derivatives - that's both unnecessary and numerically fragile
 * (grazing angles, sharply-varying normals) for every neural material
 * regardless of whether its source ever used that property at all. This was
 * a real regression (see git history) that showed up as black/NaN pixels
 * concentrated wherever the (unrelated, correctly-reconstructed) normal
 * channel varied most, which made it look like a normal-reconstruction bug.
 * Skipping the assignment entirely when the value is the channel's own
 * default keeps every `use*` gate exactly as it would be on an untouched
 * `MeshPhysicalNodeMaterial`, for every channel this pattern could affect,
 * not just the one that happened to get noticed first.
 */
for ( const channel of CHANNELS ) {

	const applyConstant = channel.applyConstant;

	channel.applyConstant = ( targetMaterial: any, constantValue: any ) => {

		if ( constantEqualsDefault( channel, constantValue ) ) return;

		applyConstant( targetMaterial, constantValue );

	};

}

// 39 as of this writing (2 of CHANNELS' 26 entries -
// `iridescenceThicknessRangeMin`/`Max` - are metadata-only and can never
// actually be "active", so this is a loose upper bound on trainable width,
// not a value every channel could simultaneously hit).
const MAX_TOTAL_CHANNELS = CHANNELS.reduce( ( sum, c ) => sum + c.size, 0 );

/**
 * Debug-only views of the mesh's raw tangent-space frame itself
 * (`tangentWorld`/`bitangentWorld`) - not trained channels, just the
 * geometry-derived basis every 'tanh'-activated normal/clearcoatNormal/
 * anisotropy channel is defined relative to. Both the teacher
 * (buildChannelPreviewMaterials) and the neural material (setDebugView)
 * expose these identically so a "normal" debug-view mismatch between the
 * two can be isolated to either the trained (dx, dy) content or this
 * shared frame - if *this* view already disagrees, the bug is in how the
 * frame itself is built/consumed, not in anything the network learned.
 */
const FRAME_VIEWS = [ 'tangent', 'bitangent' ];

function getChannel( key: string, channels: NTCChannel[] = CHANNELS ): NTCChannel {

	const channel = channels.find( ( c ) => c.key === key );
	if ( channel === undefined ) throw new Error( `THREE.NTCFormat: unknown channel "${key}".` );

	return channel;

}

/**
 * Decodes a `constantValues` map (see `NTCSource.
 * classifyMaterialChannels`, and the manifest round trip in
 * NTCManifest.js/NTCLoader.js) back into each
 * channel's true semantic values, via each channel's own (optional)
 * `decodeConstant` hook - a channel without one is passed through unchanged.
 * Currently only `attenuationDistance` defines one (unwrapping its
 * `ATTENUATION_DISTANCE_INFINITY_SENTINEL` encoding back into a real
 * `Infinity`), but this is deliberately as generic/open as every other piece
 * of this file: a caller-supplied custom channel can define its own
 * `decodeConstant` for the same reason it can define its own `applyConstant`
 * - some other physical quantity that isn't itself JSON/shader-safe in its
 * natural range.
 *
 * Unknown keys in `constantValues` (not present in `channels`) are passed
 * through unchanged rather than dropped or erroring - this only *decodes*
 * values it recognizes, it doesn't validate the map's shape (that's
 * `NTCLoader.validateManifest`'s job, for the manifest path).
 */
function decodeConstantValues( constantValues: Record<string, any>, channels: NTCChannel[] = CHANNELS ): Record<string, any> {

	const decoded: Record<string, any> = {};

	for ( const [ key, value ] of Object.entries( constantValues ) ) {

		const channel = channels.find( ( c ) => c.key === key );
		decoded[ key ] = ( channel && channel.decodeConstant ) ? channel.decodeConstant( value ) : value;

	}

	return decoded;

}

/**
 * Assigns contiguous flat offsets (and a total/pack count) to an arbitrary
 * subset of channel descriptors, in the order given - used both for the
 * "active, trained" subset (network output layout) and, incidentally, for
 * computing the full layout when every channel is active. Fully generic:
 * operates on any array of `{ key, size, activation }` objects, whether
 * that's (a subset of) the built-in `CHANNELS` or an entirely caller-
 * supplied channel array.
 */
function layoutChannels( channelSubset: NTCChannel[] ): { channels: NTCLayoutChannel[]; totalChannels: number; packCount: number } {

	let offset = 0;
	const layout: NTCLayoutChannel[] = [];

	for ( const channel of channelSubset ) {

		layout.push( { ...channel, offset } );
		offset += channel.size;

	}

	const totalChannels = offset;

	return { channels: layout, totalChannels, packCount: Math.ceil( totalChannels / 4 ) };

}

/**
 * Flattens an (offset-assigned) active-channel layout - see `layoutChannels`
 * - into a per-output-component `activation` array, e.g. for
 * `NTCTrainer`'s `channelActivations` option (see
 * NTCGPUComputeTSL.js) and for `NTCNodeMaterial`'s
 * `sliceChannels`/`applyChannelActivation` at inference time - both index
 * this the same way `layoutChannels` assigns offsets, so a component's
 * activation always lands at the same flat index its value does.
 */
function buildChannelActivations( activeChannels: NTCChannel[] ): ( NTCActivation )[] {

	const activations: NTCActivation[] = [];

	for ( const channel of activeChannels ) {

		for ( let i = 0; i < channel.size; i ++ ) activations.push( channel.activation );

	}

	return activations;

}

/**
 * Turns one channel's raw value into an unlit RGB preview color, for a
 * per-channel "debug view" - shared between the teacher (raw physical
 * value, needs remapping into display space to match) and the trained
 * network (whose output already came through this same channel's
 * `activation`, so `alreadyEncoded: true` skips the extra remap). Scalars
 * are broadcast to grayscale; 'tanh' vectors (normals, anisotropy) map to
 * the familiar bluish normal-map-style preview.
 */
function previewColor( valueNode: any, channel: { activation?: NTCActivation; size: number }, alreadyEncoded: boolean ): any {

	let value = valueNode;

	if ( channel.activation === 'tanh' && ! alreadyEncoded ) value = value.mul( 0.5 ).add( 0.5 );

	// A trained (as opposed to measured-constant) value has no guarantee of
	// landing in [0,1] - even a sigmoid/tanh-activated channel only
	// approaches its bound asymptotically, and softplus (emission) is
	// unbounded above by design. Clamp for display so this reads as "not
	// converged yet" (a muted, but plausible, color) rather than
	// hard-saturating to a misleadingly bold primary color.
	value = value.clamp( 0, 1 );

	if ( channel.size === 1 ) return vec3( value, value, value );
	if ( channel.size === 2 ) return vec3( value.x, value.y, 0 );

	return value;

}

/**
 * Builds one channel's unlit debug-view color node - widening a `type:
 * 'normal'` channel's described size to 3 first, since by the time a value
 * reaches here it's always the fully TBN-reconstructed 3-component vector
 * (whether trained-and-reconstructed or the [0,0,1] constant fallback), not
 * the raw 2-component (dx, dy) the network itself trains against - feeding
 * the real 2-component channel descriptor would hit `previewColor`'s
 * `size === 2` branch, which hardcodes blue to 0 and silently discards z.
 * (See `OUTPUT_TYPES.normal.previewSize` in NTCOutputTypes.js.) Shared
 * between `NTCSource.buildChannelPreviewMaterials` (the teacher
 * side) and `NTCNodeMaterial.setDebugView` (the neural side) so
 * the two can't drift apart on this widening.
 */
function buildDebugViewColorNode( channel: NTCChannel, valueNode: any ): any {

	const outputType = OUTPUT_TYPES[ channelEffectiveType( channel ) ];
	const previewChannel = outputType?.previewSize ? { ...channel, size: outputType.previewSize } : channel;

	return vec4( previewColor( valueNode, previewChannel, false ), 1 );

}

/**
 * Builds the debug-view color node for one of `FRAME_VIEWS` - the mesh's
 * raw tangent-space frame itself, not a trained channel. Shared for the
 * same reason as `buildDebugViewColorNode` above.
 */
function buildFrameViewColorNode( frameNode: any ): any {

	return vec4( previewColor( transformNormalToView( frameNode ), { activation: 'tanh', size: 3 }, false ), 1 );

}

/**
 * The per-material UV transform every `.ntc` model carries: a 2D affine map
 * from mesh/query UV into the local space the trained feature grids + MLP
 * were actually fit against (see `NTCNodeMaterial.js`'s query-time coord
 * build, and `training/NTCTextureSource.js`'s inverse-transformed bake).
 * Stored/serialized as 6 numbers - the affine part of a `THREE.Matrix3`
 * (`[a, b, c, d, e, f]` such that `local.x = a*u + b*v + c`, `local.y = d*u +
 * e*v + f`; the matrix's third row is always the implicit `[0, 0, 1]`) rather
 * than the full 9-element `Matrix3`, since a UV transform can never carry a
 * projective (non-affine) third row - see `encodeUvTransform`/
 * `decodeUvTransform` below. Defaults to identity - a material with no
 * detected/explicit UV transform behaves exactly as before this field
 * existed.
 */
function decodeUvTransform( array?: number[] | null ): any {

	if ( ! array ) return new THREE.Matrix3();

	const [ a, b, c, d, e, f ] = array;
	return new THREE.Matrix3().set( a, b, c, d, e, f, 0, 0, 1 );

}

/**
 * Inverse of `decodeUvTransform` - reads the affine part (row 1: `[a, b,
 * c]`, row 2: `[d, e, f]`) out of a `THREE.Matrix3`'s column-major
 * `elements` array (`[n11, n21, n31, n12, n22, n32, n13, n23, n33]`, see
 * `Matrix3`'s own doc comment on row-/column-major storage).
 */
function encodeUvTransform( matrix3: any ): number[] {

	const e = matrix3.elements;
	return [ e[ 0 ], e[ 3 ], e[ 6 ], e[ 1 ], e[ 4 ], e[ 7 ] ];

}

/**
 * True for an identity (or missing) UV transform - used to keep a `.ntc`
 * manifest's `uvTransform` field additive/optional (omitted entirely for
 * the common case where nothing was detected/supplied) rather than always
 * present, matching how `renderFlags` was introduced without a `VERSION`
 * bump (see NTCManifest.js's `encodeNTC`).
 */
function isIdentityUvTransform( matrix3?: any ): boolean {

	if ( ! matrix3 ) return true;

	const e = matrix3.elements;
	return e[ 0 ] === 1 && e[ 1 ] === 0 && e[ 2 ] === 0 &&
		e[ 3 ] === 0 && e[ 4 ] === 1 && e[ 5 ] === 0 &&
		e[ 6 ] === 0 && e[ 7 ] === 0 && e[ 8 ] === 1;

}

export {
	FORMAT,
	VERSION,
	CHANNELS,
	MAX_TOTAL_CHANNELS,
	FRAME_VIEWS,
	getChannel,
	constantEqualsDefault,
	decodeConstantValues,
	layoutChannels,
	buildChannelActivations,
	previewColor,
	buildDebugViewColorNode,
	buildFrameViewColorNode,
	simpleScalarChannel,
	fixedRangeScalarChannel,
	colorIntensityChannel,
	simpleColorChannel,
	normalChannel,
	iridescenceThicknessRangeChannel,
	constantToNode,
	reconstructFinalNormal,
	decodeUvTransform,
	encodeUvTransform,
	isIdentityUvTransform
};
