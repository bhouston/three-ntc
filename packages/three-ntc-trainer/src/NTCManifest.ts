// Exported as .ntc (JSON content, format: 'three-ntc') - FORMAT/VERSION
// live in the sibling `three-ntc` runtime package, since NTCLoader (runtime,
// no training dependencies) needs them too.

import { FORMAT, VERSION, encodeUvTransform, isIdentityUvTransform } from 'three-ntc';
import { encodeUint8Base64, encodeUint4Base64, encodeMLPLayersBase64 } from 'three-ntc';
import { computeLatentRanges, type GridLevelLayout } from './NTCQuantization.js';

/**
 * A `.ntc` (Neural Texture Compression) asset is one shared mip pyramid of
 * latent feature grids + MLP decoder (NVIDIA NTC style: one small decoder,
 * many jointly-fit, correlated PBR output channels), plus the channel
 * layout/constant-value metadata needed to slice that decoder's output back
 * into named PBR channels at load time (see NTCLoader.js / NTCNodeMaterial.js).
 *
 * `cpuModel` is `{ channels, levels, mipsPerLevel, grids, decoder: { layers },
 * outputChannels, textureResolution, maxLod }` - exactly what
 * `NTCTrainer`/`NTCFit.fitNTCMaterial()` produces (see
 * NTCGridPyramidModel.js). `channelClassification` is `{ activeChannels,
 * constantValues, totalChannels, packCount, renderFlags }` - the layout that
 * model's output was trained against (see `NTCSource.classifyMaterialChannels`).
 *
 * `options.uvTransform` (or, if absent, `cpuModel.uvTransform`) is an
 * optional `THREE.Matrix3` mapping mesh/query UV into the local space this
 * model's grids + MLP were fit against - see `NTCFormat.js`'s
 * `decodeUvTransform`/`encodeUvTransform`/`isIdentityUvTransform` and
 * `NTCNodeMaterial.js`'s query-time coord build. Omitted from the manifest
 * (defaulting to identity on load) when identity or not supplied.
 *
 * Only the channel *keys* need to be persisted, in layout order - every
 * other field on an `activeChannels` entry (`size`, `activation`,
 * `nodeKeys`, `clampRange`, `defaultValue`, `offset`) is a fixed property of
 * that key already declared in `NTCFormat.CHANNELS`/`getChannel`, and
 * `offset`/`totalChannels`/`packCount` are re-derived deterministically from
 * the key list alone via `layoutChannels` - see NTCLoader.js. Storing only
 * the keys means this manifest can never disagree with NTCFormat.js about
 * what a channel's own size/activation/clampRange is.
 */
function encodeNTC( cpuModel: any, channelClassification: any, options: any = {} ): any {

	const ranges = resolveQuantizationRanges( cpuModel, options );
	const uvTransform = options.uvTransform || cpuModel.uvTransform;
	// Latent bit depth follows the QAT mode the model was trained with;
	// 'none' (no QAT) keeps the pre-existing uint8 default.
	const dtype = cpuModel.quantization?.mode === 'uint4' ? 'uint4' : 'uint8';
	const encodeLatents = dtype === 'uint4' ? encodeUint4Base64 : encodeUint8Base64;

	const levels = cpuModel.grids.map( ( grid: any, index: number ) => {

		const [ min, max ] = ranges[ index ];

		return {
			width: grid.width,
			height: grid.height,
			channels: grid.channels,
			wrap: options.wrap || 'repeat',
			dtype,
			min,
			max,
			dataBase64: encodeLatents( grid.data, min, max )
		};

	} );

	return {
		format: FORMAT,
		version: VERSION,
		name: options.name,
		source: options.source || 'THREE.NTCNodeMaterial',
		latents: {
			channelsPerLevel: cpuModel.channels,
			wrap: options.wrap || 'repeat',
			levels,
			// Required mip-pyramid metadata (see NTCGridPyramidModel.js /
			// NTCMipBands.js) - `mipsPerLevel` + the stored level count
			// (`levels.length`) determine which physical mip a given LOD maps
			// onto, and `maxLod` is the total physical mip range this model
			// was trained to support. Without these a loader can't correctly
			// reconstruct the decoder's LOD input at all, which is why this is
			// a `VERSION` bump (2) rather than an optional/additive field like
			// most other manifest additions - see NTCFormat.js.
			mipsPerLevel: cpuModel.mipsPerLevel,
			maxLod: cpuModel.maxLod,
			// Optional/additive (default false when absent, matching every
			// manifest saved before this field existed) - see
			// NTCGridPyramidModel.js's `computeDecoderInputSize` doc comment.
			// Only written when true, since false is the default a loader
			// already falls back to.
			positionalEncoding: cpuModel.positionalEncoding || undefined
		},
		outputChannels: cpuModel.outputChannels,
		// Omitted entirely (rather than always written as the 6-number
		// identity) for an identity/absent transform - see
		// NTCFormat.isIdentityUvTransform's doc comment on why this is an
		// additive/optional field rather than a VERSION bump.
		uvTransform: ( uvTransform && ! isIdentityUvTransform( uvTransform ) ) ? encodeUvTransform( uvTransform ) : undefined,
		mlp: encodeMLPLayersBase64( cpuModel.decoder.layers ),
		// See NTCSource.resolveRenderFlags's doc comment - `side`/
		// `transparent` aren't channels (nothing for the network to fit), but
		// still need to round-trip so a loaded material's transmission pass
		// count (and therefore its attenuation tint strength) matches the
		// source material it was fit against. `undefined` on a
		// classification built without a source material (e.g. hand-assembled
		// constant-only classification) round-trips as a plain `null`.
		renderFlags: channelClassification.renderFlags || null,
		channels: {
			activeKeys: channelClassification.activeChannels.map( ( channel: any ) => channel.key ),
			constantValues: channelClassification.constantValues
		}
	};

}

/**
 * Resolves the per-level `[min, max]` quantization range used to encode the
 * exported latent grid as uint8: an explicit `options.quantizationRanges`,
 * else `cpuModel.quantizationRange` (set by `NTCTrainer.train()` when
 * quantization-aware training - QAT - was actually used, see
 * NTCQuantization.js), else a plain min/max scan over the model's *final*
 * (post-training) latents via `computeLatentRanges`.
 *
 * Note: scanning a range at export time for a model that was never trained
 * with QAT still produces a numerically valid uint8-quantized manifest, but
 * the network's weights were never exposed to the rounding error that
 * quantization introduces - so the exported model may reconstruct slightly
 * less accurately than one actually trained with `quantization.mode:
 * 'uint8'` (see NTCTrainer.js's `quantization` option). Prefer training
 * with QAT enabled when export-time compactness matters.
 */
function resolveQuantizationRanges( cpuModel: any, options: any ): Array<[ number, number ]> {

	if ( options.quantizationRanges ) return options.quantizationRanges;
	if ( cpuModel.quantizationRange ) return cpuModel.quantizationRange;

	const { flat, gridLevels } = concatenateGridData( cpuModel.grids );

	return computeLatentRanges( flat, gridLevels, true );

}

/**
 * Concatenates every grid level's flat `Float32Array`/`Array` `data` into a
 * single flat `Float32Array`, alongside the `{ offset, floatCount }` layout
 * `NTCQuantization.computeLatentRanges` expects (the same shape
 * `NTCGPUModel.computeTextureModelLayout`'s `layout.gridLevels` already
 * uses) - so a plain export-time min/max scan can reuse that shared helper
 * instead of re-deriving its own reduction loop.
 */
function concatenateGridData( grids: any[] ): { flat: Float32Array; gridLevels: GridLevelLayout[] } {

	let offset = 0;
	const gridLevels: GridLevelLayout[] = [];

	for ( const grid of grids ) {

		gridLevels.push( { offset, floatCount: grid.data.length } );
		offset += grid.data.length;

	}

	const flat = new Float32Array( offset );

	for ( let g = 0; g < grids.length; g ++ ) {

		flat.set( grids[ g ].data, gridLevels[ g ].offset );

	}

	return { flat, gridLevels };

}

export { FORMAT, VERSION, encodeNTC };
