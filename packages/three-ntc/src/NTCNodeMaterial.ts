import * as THREE from 'three';
// `MeshPhysicalNodeMaterial` only exists in the WebGPU/node-material build,
// not the base 'three' entrypoint - see `three-shims.d.ts`.
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { bitangentWorld, fract, log, max, min, step, tangentWorld, uniform, uv, vec2, vec3, vec4 } from 'three/tsl';
import { buildLevelTextures, evaluateNeuralTextureRaw, NTCCpuModel } from './NTCDecoderTSL.js';
import { applyChannelActivation } from './NTCOutputActivations.js';
import { CHANNELS, FRAME_VIEWS, getChannel, buildDebugViewColorNode, buildFrameViewColorNode, NTCChannel, NTCLayoutChannel } from './NTCFormat.js';
import { constantToNode, reconstructFinalNormal } from './NTCOutputTypes.js';

/** What `NTCLoader.parse()` / a live trainer's classification step produces. */
export interface NTCChannelClassification {
	activeChannels: NTCLayoutChannel[];
	constantValues: Record<string, any>;
	totalChannels?: number;
	packCount?: number;
	renderFlags?: { side?: any; transparent?: boolean } | null;
}

/** Options accepted by `NTCNodeMaterial`'s constructor. */
export interface NTCNodeMaterialOptions {
	channels?: NTCChannel[];
	renderer?: any | null;
	interpolation?: boolean;
	uvTransform?: any;
	lodNode?: any;
	lodBias?: number | any;
	debugView?: string;
}

/**
 * Slices the raw per-channel output array (see evaluateNeuralTextureRaw)
 * into a { [channelKey]: TSL node } map, following the offsets of the
 * *active* (trained) channel layout passed to the constructor - which,
 * unlike the full channel vocabulary, only covers whatever subset of
 * channels this particular material actually varies spatially (see
 * NTCSource.classifyMaterialChannels). Each channel's raw
 * (linear-decoder) slice is passed through its own output activation here -
 * see NTCFormat.js / ./NTCOutputActivations.js - so every
 * consumer downstream of `sliceChannels` already sees values in the
 * channel's natural physical range.
 */
function sliceChannels( outputs: any[], activeChannels: NTCLayoutChannel[] ): Record<string, any> {

	const slices: Record<string, any> = {};

	for ( const channel of activeChannels ) {

		const values = [];
		for ( let i = 0; i < channel.size; i ++ ) values.push( applyChannelActivation( outputs[ channel.offset + i ], channel.activation ) );

		if ( channel.size === 1 ) slices[ channel.key ] = values[ 0 ];
		else if ( channel.size === 2 ) slices[ channel.key ] = vec2( ...values );
		else slices[ channel.key ] = vec3( ...values );

	}

	return slices;

}

/**
 * Estimates a screen-space LOD (mip index) node from `coord`'s own
 * screen-space derivatives, the same way hardware texture filtering picks a
 * mip level - `max(|d(coord)/dx|, |d(coord)/dy|)` in texel units, log2'd.
 * Deliberately derived from `coord` (the *pre*-`fract()` tiled UV, still
 * continuous across a repeat-wrapped tile boundary) rather than the final
 * wrapped UV fed to `texture()` - `fract()`'s wraparound discontinuity would
 * otherwise show up as a spurious huge derivative (and therefore a spurious
 * max-LOD spike) at every tile seam.
 *
 * Texel-unit scaling uses `2^maxLod` as a stand-in for the source texture's
 * actual resolution: `maxLod` is derived from it as `ceil(log2(
 * textureResolution))` (see NTCGridPyramidModel.js), so `2^maxLod` is always
 * within a factor of 2 of the true resolution - close enough for a screen-
 * space LOD heuristic, and it means this only needs `maxLod`, which (unlike
 * the source texture's exact pixel size) is always available, including for
 * a model loaded from a `.ntc` file (see NTCLoader.js).
 *
 * `lodBias` (mip levels, default 0) is subtracted from the raw footprint-
 * derived estimate before clamping - the same sign convention as WebGL/
 * WebGPU's own sampler LOD bias (positive lowers the *reconstructed* LOD,
 * i.e. keeps a finer/higher-resolution stored level in use for longer as
 * the surface recedes; negative pushes toward coarser levels sooner). This
 * addon's own reconstruction can afford to stay fine longer than ordinary
 * mipmapped textures would: since the last stored level absorbs every LOD
 * past its own band as an open-ended tail (see NTCMipBands.js), aliasing
 * from under-blurring at a moderate positive bias is bounded by the MLP's
 * own reconstruction, not by an unfiltered raw texture read - a positive
 * bias here is a legitimate quality/aliasing trade-off, not just a hack.
 */
function computeAutoLodNode( coord: any, maxLod: number, lodBias: number | any = 0 ): any {

	const texelCoord = coord.mul( Math.pow( 2, maxLod ) );
	const footprint = max( texelCoord.dFdx().length(), texelCoord.dFdy().length() ).max( 1e-6 );

	return log( footprint ).div( Math.LN2 ).sub( lodBias ).clamp( 0, maxLod );

}

/**
 * A synthetic "UV checker" color node for the 'textureUv' debug view (see
 * setDebugView) - no texture asset needed. A red/green gradient across each
 * unit tile (`vec3(u, v, 0.25)`) makes the local space's orientation legible
 * at a glance - e.g. a correctly detected 90deg `rotate2d` should visibly
 * swap which mesh edge reads red vs green versus an identity transform -
 * and `fract()`'s wraparound at the query-time transform (see the
 * constructor's `tiledUV`) already turns any tiling/repeat baked into
 * `uvTransform` into repeated ramps across the surface. The overlaid dark
 * grid (10 cells per tile) makes both the tile boundaries and any shear/
 * non-uniform scale in the transform easier to read than the gradient
 * alone would.
 */
function buildTextureUvDebugColorNode( uvNode: any ): any {

	const cell = fract( uvNode.mul( 10 ) );
	const distanceToNearestGridLine = min( cell, cell.oneMinus() );
	const lineHalfWidth = 0.04;
	const onGridLine = max(
		step( distanceToNearestGridLine.x, lineHalfWidth ),
		step( distanceToNearestGridLine.y, lineHalfWidth )
	);

	const gradient = vec3( uvNode.x, uvNode.y, 0.25 );
	const color = gradient.mix( vec3( 0 ), onGridLine );

	return vec4( color, 1 );

}

/**
 * A `MeshPhysicalNodeMaterial` driven by one trained `NTCTrainer`
 * model: channels the source material actually varied spatially (`
 * activeChannels`) read a decoded slice of one shared grid + MLP forward
 * pass - all from the *same* pass, matching the NVIDIA neural texture
 * compression "one decoder, many correlated channels" design - while
 * channels that were just a flat constant on the source material
 * (`constantValues`) are applied directly as ordinary material properties,
 * bypassing the network entirely (see NTCSource.
 * classifyMaterialChannels for why: no sense spending network capacity, or
 * training-sample budget, reproducing a value that never varies).
 *
 * Applying a trained slice or a resolved constant onto this material is
 * itself channel-defined, not hardcoded here - see each channel descriptor's
 * `applyActive`/`applyConstant` in NTCFormat.js. This constructor
 * is a fully generic loop over whatever `channels` array the classification
 * was built from (`options.channels`, defaulting to the built-in `CHANNELS`
 * vocabulary), so a caller-supplied custom channel array works exactly the
 * same way the built-in one does.
 *
 * @three_import import { NTCNodeMaterial } from 'three/addons/ntc/NTCNodeMaterial.js';
 */
class NTCNodeMaterial extends ( MeshPhysicalNodeMaterial as any ) {

	cpuModel: NTCCpuModel;
	activeChannels: NTCLayoutChannel[];
	channels: NTCChannel[];
	interpolation: boolean;
	mipChainTexture: any;
	levelTextures: any[] | null;
	uvTransform: any;

	private _localUv: any;
	private _slices: Record<string, any>;
	private _constantValues: Record<string, any>;
	private _shadedColorNode: any;
	private _lodBiasUniform: any;

	/**
	 * `channelClassification` is whatever `NTCSource.
	 * classifyMaterialChannels` returned - `{ activeChannels, constantValues,
	 * totalChannels, packCount }` - passed through as-is rather than
	 * destructured by the caller, since the two fields this constructor
	 * needs are always produced together and never meaningfully used apart.
	 *
	 * `options.channels` is the full channel array (active + constant alike)
	 * `channelClassification` was produced from - defaults to the built-in
	 * `CHANNELS` vocabulary, matching `classifyMaterialChannels`'s own
	 * default, since `setDebugView` needs every channel descriptor (not just
	 * the active ones) to resolve a debug view by key.
	 *
	 * `options.renderer` is accepted for callers that already pass one through
	 * from examples, but the stock path uses fp32 decoder uniforms.
	 */
	// `options.lodNode` overrides the LOD (mip index) this material
	// reconstructs at - a TSL float node, useful for an explicit distance-
	// based LOD or for debugging a specific mip level. Defaults to an
	// automatic screen-space-derivative estimate (`computeAutoLodNode`
	// above), the same way hardware mipmapping picks a level - which is what
	// makes this material anti-alias correctly when viewed from a distance
	// instead of always reconstructing at full (finest-LOD) detail.
	//
	// `options.lodBias` (mip levels, default 0) only applies to that
	// automatic estimate (ignored when `options.lodNode` is supplied) - see
	// `computeAutoLodNode`'s doc comment. A positive value delays the switch
	// to coarser stored levels as a surface recedes; e.g. `lodBias: 1` keeps
	// this material one full mip level finer than the raw screen-space
	// estimate at every distance. May be a plain JS number (baked into the
	// node graph at construction time) or a TSL node (e.g. `uniform(1)`) -
	// passing a `uniform()` node lets a live GUI control (see
	// webgpu_materials_neural_texture_compression.html) retune this after
	// the material's already built, by writing `.value` on that same node,
	// shared across every material constructed with it, with no rebuild.
	constructor( cpuModel: NTCCpuModel, channelClassification: NTCChannelClassification, options: NTCNodeMaterialOptions = {} ) {

		super();

		const { activeChannels, constantValues, renderFlags } = channelClassification;
		const channels = options.channels || CHANNELS;

		// Mirror the source material's `side`/`transparent` (see
		// NTCSource.resolveRenderFlags's doc comment) - most
		// importantly so a transmissive source's `DoubleSide` survives onto
		// this material too, keeping its Beer-Lambert attenuation pass count
		// (and therefore tint strength) consistent with what it was fit
		// against. `renderFlags` is undefined for a manifest saved before this
		// field existed - falls back to this class's inherited default
		// (`FrontSide`/opaque) exactly as before.
		if ( renderFlags ) {

			if ( renderFlags.side !== undefined ) this.side = renderFlags.side;
			if ( renderFlags.transparent !== undefined ) this.transparent = renderFlags.transparent;

		}

		this.cpuModel = cpuModel;
		this.activeChannels = activeChannels;
		this.channels = channels;
		// See buildMipChainTexture's doc comment / setInterpolation below -
		// `false` swaps to nearest-neighbor sampling *within* each stored mip
		// level (still blending *between* levels) so the trained feature
		// grid's actual texels can be inspected without bilinear blur hiding
		// them. Not part of the trained model or the `.ntc` format - a pure
		// display-time sampler setting, so toggling it later (setInterpolation)
		// never needs a reload/retrain.
		this.interpolation = options.interpolation !== false;
		// `positionalEncoding` models (see NTCDecoderTSL.js's
		// `evaluatePositionalEncodingFeatures`) fetch raw texels from one
		// `DataTexture` per stored level instead of sampling the mip-chain
		// texture - build only whichever this model actually needs.
		this.mipChainTexture = null;
		this.levelTextures = buildLevelTextures( cpuModel );

		// Maps mesh/query UV into the local space this model's grids + MLP
		// were actually fit against - `options.uvTransform` overrides
		// `cpuModel.uvTransform` (the transform persisted with/detected for
		// this model, see NTCFormat.js's decodeUvTransform and
		// training/NTCTextureSource.js's inverse-transformed bake), both
		// defaulting to identity. Uploaded as a `mat3` uniform and applied
		// the same way `TextureNode.getTransformedUV` applies `texture.
		// matrix` - `matrix.mul(vec3(uv,1)).xy` - rather than a bespoke
		// scale+offset pair, so this material can express the same rotate/
		// scale/offset-around-a-pivot transforms `THREE.Texture`/MaterialX's
		// `place2d` already support.
		this.uvTransform = options.uvTransform || cpuModel.uvTransform || new THREE.Matrix3();
		const uvTransformUniform = uniform( this.uvTransform );

		const coord = uvTransformUniform.mul( vec3( uv(), 1 ) ).xy;
		const tiledUV = fract( coord );

		// Stashed for the 'textureUv' debug view (see setDebugView) - exactly
		// the UV this material actually queries the feature grid + MLP with,
		// so that view doubles as a visual check on whether `this.uvTransform`
		// (detected or explicit) is correct.
		this._localUv = tiledUV;

		// Auto-LOD (see computeAutoLodNode above) needs `coord` (pre-`fract()`,
		// still continuous across a tile seam) - built unless the caller
		// already supplied an explicit override. `lodBias` is always wrapped in
		// our own `uniform()` node (seeded from a plain-number `options.
		// lodBias`, or from a node's current `.value` if one was passed) so
		// `setLodBias` below can retune it live, with no material rebuild -
		// exactly the upgrade path this option's own doc comment describes.
		if ( ! options.lodNode ) {

			const initialBias = typeof options.lodBias === 'number' ? options.lodBias : options.lodBias?.value ?? 0;
			this._lodBiasUniform = uniform( initialBias );

		}

		const lodNode = options.lodNode || computeAutoLodNode( coord, cpuModel.maxLod, this._lodBiasUniform );

		const outputs = evaluateNeuralTextureRaw( tiledUV, cpuModel, this.mipChainTexture, options.renderer || null, lodNode, this.levelTextures );
		const slices = sliceChannels( outputs, activeChannels );
		this._slices = slices;
		this._constantValues = constantValues;

		const isActive = ( key: string ) => Object.prototype.hasOwnProperty.call( slices, key );

		// Every channel in the vocabulary is applied identically here: a
		// trained (active) channel's decoded slice goes through its own
		// `applyActive`, an untrained (constant, or `alwaysConstant`) channel's
		// resolved value goes through its own `applyConstant` - see each
		// descriptor in NTCFormat.js for what that actually does
		// (assign a *Node property, set a plain property, decompose into a
		// legacy strength/rotation pair, leave a property untouched, ...).
		for ( const channel of channels ) {

			if ( isActive( channel.key ) ) channel.applyActive( this, slices[ channel.key ] );
			else channel.applyConstant( this, constantValues[ channel.key ] );

		}

		// `albedo`'s applyActive stashes its trained colorNode as
		// `_shadedColorNode` (see NTCFormat.js) - only set when
		// albedo is actually active; a constant-albedo material has no
		// "trained color node" to swap back in via setDebugView('shaded').
		this._shadedColorNode = this._shadedColorNode || null;

		this.setDebugView( options.debugView || 'shaded' );

	}

	/**
	 * Swaps the visible output between the full physically-shaded material
	 * and an unlit preview of one channel in isolation - trained or
	 * constant alike - matching `NTCSource.
	 * buildChannelPreviewMaterials` on the teacher side, so both are
	 * comparable directly.
	 */
	setDebugView( view: string ): void {

		this.userData.debugView = view;

		if ( view === 'shaded' ) {

			this.lights = true;
			this.toneMapped = true;
			if ( this._shadedColorNode ) ( this as any ).colorNode = this._shadedColorNode;

		} else if ( FRAME_VIEWS.includes( view ) ) {

			// Debug-only views of the raw tangentWorld/bitangentWorld frame
			// itself, bypassing the trained network entirely - see
			// FRAME_VIEWS's doc comment in NTCFormat.js and
			// NTCSource.buildChannelPreviewMaterials's matching
			// 'tangent'/'bitangent' entries. Both this mesh and the teacher
			// mesh share the same geometry (and therefore the same tangent/
			// bitangent vertex attribute), so this view should be pixel-
			// identical between the two - if it isn't, the mismatch is in
			// how the frame is built/consumed here, not in anything the
			// network learned.
			this.lights = false;
			this.toneMapped = false;
			const frameNode = view === 'tangent' ? tangentWorld : bitangentWorld;
			( this as any ).colorNode = buildFrameViewColorNode( frameNode );

		} else if ( view === 'textureUv' ) {

			// Debug-only, neural-material-only (see this module's export
			// comment and webgpu_materials_neural_texture_compression_
			// trainer.html's viewFolder options) - not a trained channel, and
			// not mirrored on the teacher side the way FRAME_VIEWS is: the
			// teacher's albedo/etc. graph has no equivalent "local UV space",
			// only whatever raw mesh UV its own image lookups use. Visualizes
			// `this._localUv` - the exact post-`uvTransform`, tiled UV this
			// material queries the feature grid + MLP with (see the
			// constructor) - so a detected/explicit `uvTransform` can be
			// judged visually: a correct one should show this grid oriented/
			// tiled exactly the way the source image's own UV space was.
			this.lights = false;
			this.toneMapped = false;
			( this as any ).colorNode = buildTextureUvDebugColorNode( this._localUv );

		} else {

			// Tone mapping (ACES etc.) is meant for real lit HDR output, not a
			// flat diagnostic color - left at its default (true) here, it gets
			// applied to this raw channel value anyway, which can skew colors
			// (filmic curves are notorious for pushing bright/uneven values
			// toward warm/red tones) and make this view uncomparable to the
			// teacher's preview material, which already sets toneMapped=false
			// (see NTCSource.buildChannelPreviewMaterials).
			this.lights = false;
			this.toneMapped = false;
			const channel = getChannel( view, this.channels );
			const active = Object.prototype.hasOwnProperty.call( this._slices, view );

			// `this._slices[ view ]` already went through this channel's own
			// output activation (see sliceChannels/applyChannelActivation), so
			// it's in the same raw physical range as the teacher's resolved
			// node - always `alreadyEncoded: false` here, exactly like
			// `NTCSource.buildChannelPreviewMaterials`'s own call,
			// so both sides remap into display space identically.
			//
			// A `type: 'normal'` channel needs one extra step: `this._slices[
			// view ]` is still the raw tangent-space (dx, dy) offset, whereas
			// the teacher's preview (NTCSource.
			// buildChannelPreviewMaterials) shows `material.normalNode`, which
			// MaterialXLoader already blends through the mesh's own TBN basis
			// into a final view-space normal (see NTCOutputTypes.
			// reconstructFinalNormal's doc comment). Comparing the raw
			// tangent-space slice against that would show two different spaces
			// side by side - route it through the same TBN blend used for
			// actual shading (via the channel's own `applyActive`, which does
			// exactly this) so both previews are apples-to-apples. Only
			// `type: 'normal'` channels need this; every other channel's
			// slice is already in its final, directly-previewable form.
			const value = active
				? ( channel.type === 'normal' ? reconstructFinalNormal( this._slices[ view ] ) : this._slices[ view ] )
				: constantToNode( this._constantValues[ view ] );

			// `channel.size` (2, for a `type: 'normal'` channel) describes the
			// *trained* (dx, dy) payload, not this preview value - for those
			// channels it's always the TBN-reconstructed, fully 3-component
			// view-space normal above (both when active and, via
			// constantToNode, in the constant-fallback case).
			// `buildDebugViewColorNode` widens the channel descriptor
			// accordingly (see OUTPUT_TYPES.normal.previewSize in
			// NTCOutputTypes.js), so all three components actually reach
			// the display color, matching the familiar blue-dominant
			// tangent-space normal map palette instead of a z-less yellow one.
			( this as any ).colorNode = buildDebugViewColorNode( channel, value );

		}

		this.needsUpdate = true;

	}

	/**
	 * Toggles whether the feature grid's mip-chain texture (`this.
	 * mipChainTexture`, see `buildMipChainTexture`) is sampled with bilinear
	 * filtering within each stored mip level (`true`, the default) or plain
	 * nearest-neighbor (`false`) - mip levels are still blended into each
	 * other either way, only the filtering *within* one level changes. Useful
	 * for visually inspecting the trained feature grid's actual stored texels
	 * (e.g. via the 'textureUv' debug view) without bilinear blur hiding them.
	 *
	 * This purely reassigns the existing texture's `minFilter`/`magFilter` -
	 * the GPU sampler these select is looked up/created by a filter-mode key
	 * (see `updateSampler` in src/renderers/webgpu/utils/WebGPUTextureUtils.
	 * js), so no rebuild of this material, its node graph, or the underlying
	 * grid/MLP data is needed; `needsUpdate` is set only to make sure a
	 * pending render picks up the change.
	 */
	setInterpolation( enabled: boolean ): void {

		// No-op for a `positionalEncoding` model: its `levelTextures` are
		// always raw/`NearestFilter` (see NTCDecoderTSL.js's
		// `evaluatePositionalEncodingFeatures` - it needs exact texel values,
		// not a filtered blend), so there is no mip-chain texture here to
		// retune.
		if ( ! this.mipChainTexture ) return;

		this.interpolation = Boolean( enabled );
		this.mipChainTexture.magFilter = this.interpolation ? THREE.LinearFilter : THREE.NearestFilter;
		this.mipChainTexture.minFilter = this.interpolation ? THREE.LinearMipmapLinearFilter : THREE.NearestMipmapLinearFilter;
		this.mipChainTexture.needsUpdate = true;

	}

	/**
	 * Retunes the auto-LOD bias (see `computeAutoLodNode`'s doc comment) after
	 * construction - just writes `.value` on the `uniform()` node the
	 * constructor already built `lodNode` from, so (unlike `lodBias`'s
	 * one-time-baked plain-number form) this takes effect on the very next
	 * frame with no material/node-graph rebuild. A no-op when this material
	 * was built with an explicit `options.lodNode` override, since there's no
	 * bias uniform feeding into a caller-supplied LOD node.
	 */
	setLodBias( bias: number ): void {

		if ( this._lodBiasUniform ) this._lodBiasUniform.value = bias;

	}

	dispose(): void {

		if ( this.mipChainTexture ) this.mipChainTexture.dispose();
		if ( this.levelTextures ) for ( const texture of this.levelTextures ) texture.dispose();

		super.dispose();

	}

}

// The end-to-end classify -> bake -> train -> reconstruct convenience path
// (formerly a `static fit()` here) lives in `training/NTCFit.js` - it pulls
// in `NTCTrainer`/`NTCSource`, which this file deliberately doesn't: every
// other export here only *decodes* an already-trained model (loader- and
// inference-example-facing), with no training-side dependencies at all.

export { NTCNodeMaterial, sliceChannels, reconstructFinalNormal };
