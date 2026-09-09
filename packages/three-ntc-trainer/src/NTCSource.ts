import { bitangentWorld, float, tangentWorld, vec4 } from 'three/tsl';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { CHANNELS, FRAME_VIEWS, layoutChannels, buildDebugViewColorNode, buildFrameViewColorNode } from 'three-ntc';
import { constantToNode } from 'three-ntc';

type TSLNode = any;
type MaterialLike = any;
type ChannelDescriptor = any;

/**
 * Resolves every channel in a channel array (see NTCFormat.
 * CHANNELS, or `NTCFormat.js`'s doc comment on the descriptor
 * shape) to its raw (physical, un-encoded) TSL node, via each channel's own
 * `resolveNode(material)`. This is a fully generic loop - it has no
 * knowledge of what any particular channel key means; that knowledge lives
 * entirely in the channel descriptor itself (see NTCFormat.js).
 * Baking, channel-preview materials, and constant-channel detection all
 * build on top of this (or, for channels classified as constant, each
 * channel's own `resolveConstant`, called from `classifyMaterialChannels`
 * below).
 *
 * A channel without its own `resolveNode` (only possible for a caller-
 * supplied `alwaysConstant` channel that skips node resolution entirely -
 * see `classifyMaterialChannels`) falls back to its resolved constant value,
 * converted to a node, so every channel still has *some* preview-able node
 * here even if it's never actually driven by the network.
 */
function resolveMaterialChannelNodes( material: MaterialLike, channels: ChannelDescriptor[] = CHANNELS ): Record<string, TSLNode> {

	const nodes: Record<string, TSLNode> = {};

	for ( const channel of channels ) {

		nodes[ channel.key ] = channel.resolveNode
			? channel.resolveNode( material )
			: constantToNode( channel.alwaysConstant ? channel.constantValue : channel.resolveConstant( material ) );

	}

	return nodes;

}

/**
 * Splits a material's channels into the ones worth training (driven by a
 * node graph - texture or procedural, so potentially spatially-varying) and
 * the ones that are just a flat constant (a plain scalar/color property,
 * with no node at all) - so the network's output width, and the cost of
 * every training sample, scale with how much of the material actually
 * varies rather than always paying for the full channel vocabulary.
 * Constant channels are applied directly as ordinary material properties at
 * reconstruction time, bypassing the network entirely (see
 * NTCNodeMaterial.js).
 *
 * This is a node-presence heuristic, not a pixel-level analysis: a channel
 * is "active" (trained) iff at least one of its `nodeKeys` is set on the
 * material, regardless of whether that node's output actually varies once
 * evaluated - a texture-driven channel that happens to be uniform everywhere
 * still gets trained (harmlessly - a constant target is trivially fit, see
 * the flat-velvet-color regression test in NTCTrainer). An earlier
 * version of this function baked and read back each candidate channel to
 * measure its real variance and reclassify accordingly; that added real
 * complexity and failure surface (a GPU readback round trip per material
 * load) without a clear net win, so it's been backed out in favor of this
 * simpler, synchronous check.
 *
 * A channel descriptor may instead set `alwaysConstant: true` with a fixed
 * `constantValue` baked in at spec-construction time, to declare a value
 * that's *never* trained and never even checks `nodeKeys` - e.g. a value a
 * caller's custom channel array wants surfaced through `constantValues`
 * (and applied via `applyConstant`) for every material, without offering any
 * node-driven path to vary it at all.
 *
 * `channels` defaults to the built-in `CHANNELS` vocabulary, but accepts any
 * caller-supplied channel array (built-in descriptors, custom ones, or a mix)
 * - this function has no hardcoded knowledge of any particular channel key.
 */
function classifyMaterialChannels( material: MaterialLike, channels: ChannelDescriptor[] = CHANNELS ) {

	const activeList: ChannelDescriptor[] = [];
	const constantValues: Record<string, unknown> = {};

	for ( const channel of channels ) {

		if ( channel.alwaysConstant ) {

			constantValues[ channel.key ] = channel.constantValue;
			continue;

		}

		const hasNode = channel.nodeKeys.some( ( key: string ) => Boolean( material[ key ] ) );

		if ( hasNode ) {

			activeList.push( channel );

		} else {

			constantValues[ channel.key ] = channel.resolveConstant( material );

		}

	}

	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );

	return { activeChannels, totalChannels, packCount, constantValues, renderFlags: resolveRenderFlags( material ) };

}

/**
 * Captures the source material's `side`/`transparent` flags so
 * `NTCNodeMaterial` can mirror them (see its constructor) instead
 * of always defaulting to `FrontSide`/opaque. This matters most for
 * transmission: MaterialX's OpenPBR conversion sets `side = DoubleSide` +
 * `transparent = true` whenever `transmission_weight` is active (see
 * `setTransmissionFlags` in `../loaders/materialx/MaterialXSurfaceMappings.js`),
 * which makes the renderer render that mesh in two passes (back-face then
 * front-face - see `RenderList.needsDoublePass`/`Renderer._renderTransparents`)
 * and therefore apply the `attenuationColor`/`attenuationDistance` Beer-
 * Lambert absorption *twice* through the volume. A neural material that
 * stays single-pass applies that same absorption only once, so its
 * attenuation tint comes out visibly weaker (roughly the square root of the
 * teacher's) even when the trained `attenuationColor`/`attenuationDistance`/
 * `thickness` channels themselves match exactly - copying these two flags
 * keeps the neural material's pass count (and therefore its tint strength)
 * consistent with the material it was fit against.
 */
function resolveRenderFlags( material: MaterialLike ): { side: unknown; transparent: boolean | undefined } {

	return { side: material.side, transparent: material.transparent };

}

function flattenChannelComponents( activeChannels: ChannelDescriptor[], channelNodes: Record<string, TSLNode> ): TSLNode[] {

	const components: TSLNode[] = [];
	const axes = [ 'x', 'y', 'z' ];

	for ( const channel of activeChannels ) {

		const value = channelNodes[ channel.key ];

		if ( channel.size === 1 ) {

			components.push( value );

		} else {

			// Multi-component channels (2-component tanh offsets, etc.) train
			// against only their first `size` components - the raw resolved
			// node may still be a wider vector (e.g. the material's tangent-
			// space normalNode, resolved as a full vec3); trailing components
			// are deliberately dropped here (and, for normal/clearcoatNormal,
			// reconstructed at consumption time instead - see
			// NTCOutputTypes.reconstructFinalNormal).
			for ( let i = 0; i < channel.size; i ++ ) {

				let component = value[ axes[ i ] ];

				// A channel may declare a per-component transform to apply
				// only at bake time (see `transformBakeComponent` on the
				// normal/clearcoatNormal descriptors in NTCFormat.js
				// - a UV/tangent-handedness artifact of the flat-quad baking
				// setup, not part of the channel's own reconstruction
				// semantics).
				if ( channel.transformBakeComponent ) component = channel.transformBakeComponent( component, i );

				components.push( component );

			}

		}

	}

	return components;

}

function packComponentsIntoVec4( components: TSLNode[] ): TSLNode[] {

	const packs: TSLNode[] = [];

	for ( let i = 0; i < components.length; i += 4 ) {

		const group = components.slice( i, i + 4 );
		while ( group.length < 4 ) group.push( float( 0 ) );
		packs.push( vec4( group[ 0 ], group[ 1 ], group[ 2 ], group[ 3 ] ) );

	}

	return packs;

}

/**
 * Builds the packed RGBA colorNodes (see NTCFormat.js) used to
 * bake training targets for exactly the given (active/trained) channels, in
 * their layout order and raw physical units - each channel's `activation`
 * (see NTCFormat.js / NTCOutputActivations.js) is what maps
 * the network's own output into this same raw range, so no target-side
 * encoding happens here.
 */
function buildPackedColorNodes( activeChannels: ChannelDescriptor[], material: MaterialLike ): TSLNode[] {

	const channelNodes = resolveMaterialChannelNodes( material, activeChannels );
	const components = flattenChannelComponents( activeChannels, channelNodes );

	return packComponentsIntoVec4( components );

}

/**
 * Bakes every active (trained) PBR channel of a material into
 * `activeChannels.packCount`-worth of RGBA textures via one fullscreen-quad
 * render per pack (see NTCFormat.js for how flat channel indices
 * map onto them).
 */
async function bakeMaterialToTextures( renderer: any, material: MaterialLike, resolution: number, activeChannels: ChannelDescriptor[], uvTransform: any = null ) {

	const colorNodes = buildPackedColorNodes( activeChannels, material );
	const renderTargets = [];

	for ( const colorNode of colorNodes ) {

		// `generateMipmaps: true` - these are training-source textures (see
		// bakeColorNodeToTexture's doc comment on its default). `uvTransform`
		// (see its doc comment there) bakes every channel in its local,
		// untransformed space - the same space `NTCNodeMaterial` maps query
		// UV into at render time.
		renderTargets.push( await bakeColorNodeToTexture( renderer, colorNode, resolution, { generateMipmaps: true, uvTransform } ) );

	}

	return renderTargets;

}

/**
 * Builds one unlit `THREE.NodeMaterial` per channel in `channels` (plus the
 * `shaded` key, which is just the original material) so a "debug view" GUI
 * control can swap the teacher mesh's material to inspect any single channel
 * in isolation - active or constant alike - mirroring
 * `NTCNodeMaterial.setDebugView` on the neural side, so both are
 * comparable in the same encoded [0,1] space. `channels` defaults to the
 * built-in `CHANNELS` vocabulary.
 */
function buildChannelPreviewMaterials( material: MaterialLike, channels: ChannelDescriptor[] = CHANNELS ): Record<string, MaterialLike> {

	const channelNodes = resolveMaterialChannelNodes( material, channels );
	const materials: Record<string, MaterialLike> = { shaded: material };

	// Every preview below is built as a *clone* of the real (MeshPhysical-
	// NodeMaterial-derived) `material`, not a bare `THREE.NodeMaterial()`.
	// That matters specifically for `bitangentWorld`: it's never a stored
	// vertex attribute (only `tangent.xyz` + a handedness sign are - see
	// TangentUtils/Bitangent.js), so it's *always* re-derived per pixel as
	// normalWorld.cross(tangentWorld) inside a `.once(['NORMAL'])`-cached
	// helper that only gets wired up to a real per-frame varying when the
	// material actually runs a normal-mapping build pass. A bare
	// `NodeMaterial` with only `colorNode` set never triggers that pass, so
	// its `bitangentWorld` was observed to freeze at its first-build value
	// instead of tracking the mesh's rotation - `tangentWorld` doesn't have
	// this problem (it comes straight off the stored vertex attribute), and
	// `NTCNodeMaterial` doesn't either, since it's a full
	// MeshPhysicalNodeMaterial subclass that always runs that setup
	// regardless of `lights`. Cloning `material` here (which already has
	// `normalNode`/`clearcoatNormalNode` set) gives every preview the same
	// class and the same setup path, so `bitangentWorld` behaves exactly as
	// it does in the real 'shaded' render and in the neural material.
	function clonePreviewMaterial() {

		const previewMaterial = material.clone();
		previewMaterial.lights = false;
		previewMaterial.toneMapped = false;
		return previewMaterial;

	}

	for ( const channel of channels ) {

		const previewMaterial = clonePreviewMaterial();
		previewMaterial.colorNode = buildDebugViewColorNode( channel, channelNodes[ channel.key ] );
		materials[ channel.key ] = previewMaterial;

	}

	// Debug-only views of the raw tangentWorld/bitangentWorld frame itself
	// (see FRAME_VIEWS's doc comment) - view-space, same tanh-vector color
	// convention as the 'normal'/'clearcoatNormal' channels above, so this
	// is directly comparable against NTCNodeMaterial's own
	// 'tangent'/'bitangent' debug views.
	const frameNodes: Record<string, TSLNode> = { tangent: tangentWorld, bitangent: bitangentWorld };

	for ( const key of FRAME_VIEWS ) {

		const previewMaterial = clonePreviewMaterial();
		previewMaterial.colorNode = buildFrameViewColorNode( frameNodes[ key ] );
		materials[ key ] = previewMaterial;

	}

	return materials;

}

export {
	bakeMaterialToTextures,
	buildPackedColorNodes,
	buildChannelPreviewMaterials,
	classifyMaterialChannels,
	resolveMaterialChannelNodes
};
