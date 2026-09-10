import { computeFeatureLodOffset } from 'three-ntc';
import { Matrix3 } from 'three';
import { POSITIONAL_ENCODING_SIZE } from 'three-ntc';
import { createMLP, type MLP } from './NTCMLP.js';
import { computeGridLevels, createLatentGrid, LATENT_INIT_SCALE, DEFAULT_MIPS_PER_LEVEL, MAX_GRID_RESOLUTION, type LatentGrid } from './NTCGridModel.js';

interface NTCGridPyramidOptions {
	// Feature-vector width per grid cell - the paper's "grid channels"
	// (Table 2, C_k in Section 4.4; the `.ntc` manifest's `channelsPerLevel`).
	// Distinct from the decoder's `outputChannels` (the texture set's channel
	// count) and from the PBR channel *vocabulary* `fitNTCMaterial`/
	// `NTCNodeMaterial` call `channels`.
	gridChannels?: number;
	lowResChannels?: number;
	levels?: number;
	baseResolution?: number;
	mipsPerLevel?: number;
	hiddenSizes?: number[];
	hiddenActivation?: string;
	outputChannels?: number;
	textureResolution?: number;
	uvTransform?: any;
	positionalEncoding?: boolean;
	/** Zero retains grid-cell phase; eight selects the paper-style texel tile. */
	positionalEncodingPeriod?: number;
	dualGrid?: boolean;
	[key: string]: unknown;
}

interface ResolvedNTCGridPyramidOptions {
	channels: number;
	lowResChannels: number;
	levels: number;
	baseResolution: number;
	mipsPerLevel: number;
	hiddenSizes: number[];
	hiddenActivation: string;
	outputChannels: number;
	textureResolution: number | undefined;
	uvTransform: any;
	positionalEncoding: boolean;
	dualGrid: boolean;
}

/** Four G0 taps (with positional encoding), a bilinear G1 vector, and LOD.
 * Every feature level has its own independently trained half-resolution G1.
 */
function computeDecoderInputSize( channels: number, positionalEncoding: boolean, dualGrid: boolean = false, lowResChannels = channels ): number {

	return ( positionalEncoding ? channels * 4 + POSITIONAL_ENCODING_SIZE : channels ) + ( dualGrid ? lowResChannels : 0 ) + 1;

}

/**
 * Single source of truth for the model-shape options shared by the CPU
 * model, the GPU training layout, and the trainer's defaults. Resolving
 * these once here - rather than each of those three re-declaring the same
 * defaults independently - means they can't silently drift apart.
 *
 * `textureResolution` (the source texture's largest dimension) determines
 * `maxLod` - the total mip range this model is meant to support, down to
 * 1x1 - independently of how many feature levels are actually *stored*
 * (`levels`/`mipsPerLevel`, see NTCGridModel.js/NTCMipBands.js): a handful of
 * stored levels can still be asked to reconstruct a much deeper mip chain,
 * each level just gets reused for more than one of those mips.
 *
 * `baseResolution` (the *finest* stored grid's resolution) defaults to
 * `textureResolution` itself (capped at `MAX_GRID_RESOLUTION`) when not given
 * explicitly - this matters: LOD 0 always means "reconstruct the source
 * texture's own mip 0", so if the finest grid were left far coarser than the
 * texture (e.g. a fixed small default while the texture is large), the model
 * would be asked to reconstruct much more detail at LOD 0 than its finest
 * grid could ever hold, and - since `levels`/`mipsPerLevel` only cover a
 * `levels * mipsPerLevel`-mip-deep band before the *last* stored level
 * becomes an open-ended tail (see NTCMipBands.js) - most realistic viewing
 * distances would land on that same tiny tail level, looking close to a flat
 * color. Tying the default finest resolution to the real texture resolution
 * keeps LOD 0 meaningful without the caller having to know to set
 * `baseResolution` explicitly. Falls back to a fixed 128 only when neither
 * `baseResolution` nor `textureResolution` is known (e.g. a model built by
 * hand, with no real texture in the picture at all).
 */
function resolveNTCGridPyramidOptions( options: NTCGridPyramidOptions = {} ): ResolvedNTCGridPyramidOptions {

	const textureResolution = options.textureResolution;

	return {
		channels: options.gridChannels || 4,
		lowResChannels: options.lowResChannels || options.gridChannels || 4,
		levels: options.levels || 4,
		baseResolution: options.baseResolution || ( textureResolution ? Math.min( textureResolution, MAX_GRID_RESOLUTION ) : 128 ),
		mipsPerLevel: options.mipsPerLevel || DEFAULT_MIPS_PER_LEVEL,
		hiddenSizes: options.hiddenSizes || [ 32, 32 ],
		// Hidden-layer activation - 'relu' (the default, cheapest on mobile:
		// see NTCMLPTSL.js's evaluateLinearLayerMat4) or 'hgelu' (the NVIDIA
		// neural texture compression paper's own cheap GELU approximation,
		// Section 4.4 - usually a quality win, at a small extra ALU cost per
		// hidden neuron; see NTCMLP.js's hardGELU doc comment). The decoder's
		// always-linear output layer is unaffected by this option.
		hiddenActivation: options.hiddenActivation || 'relu',
		outputChannels: options.outputChannels || 3,
		textureResolution,
		// The mesh/query-UV-to-local-space affine transform this model is
		// meant to be queried through (see NTCNodeMaterial.js) - defaults to
		// identity, carried on `cpuModel.uvTransform` all the way through
		// export (NTCManifest.js) so a caller never has to re-supply it.
		uvTransform: options.uvTransform || new Matrix3(),
		// Optional (default off) - see computeDecoderInputSize's doc comment.
		positionalEncoding: options.positionalEncoding === true,
		dualGrid: options.dualGrid === true

	};

}

interface NTCGridPyramidModel {
	channels: number;
	lowResChannels: number;
	levels: number;
	mipsPerLevel: number;
	resolutions: number[];
	grids: LatentGrid[];
	lowResGrids: LatentGrid[];
	decoder: MLP;
	hiddenSizes: number[];
	hiddenActivation: string;
	outputChannels: number;
	textureResolution: number;
	maxLod: number;
	lodOffset: number;
	positionalEncodingPeriod: number;
	uvTransform: any;
	quantizationRange?: Array<[ number, number ]> | null;
	quantization?: { mode: string };
	positionalEncoding: boolean;
	dualGrid: boolean;
}

/**
 * Creates the CPU-side reference model: a genuine mip pyramid of feature
 * grids (`resolutions`/`grids`, finest-first - see `computeGridLevels`) plus
 * a small MLP decoder.
 *
 * At any given LOD, exactly one grid level's `channels`-wide feature vector
 * (selected via `NTCMipBands.selectFeatureLevel` - not concatenated with any
 * other level's) feeds the decoder, alongside the normalized LOD itself -
 * which the decoder needs because a single stored level is reused to
 * reconstruct several different physical mips (see NTCMipBands.js's doc
 * comment). This is what makes the decoder's input a fixed `channels + 1`
 * wide regardless of how many mip levels the pyramid stores, and it mirrors
 * the NVIDIA neural texture compression paper's own decoder input (Section
 * 4.4: one feature level's taps plus a LOD value).
 */
function createNTCGridPyramidModel( options: NTCGridPyramidOptions, random: () => number ): NTCGridPyramidModel {

	const { channels, lowResChannels, levels: requestedLevels, baseResolution, mipsPerLevel, hiddenSizes, hiddenActivation, outputChannels, textureResolution, uvTransform, positionalEncoding, dualGrid } = resolveNTCGridPyramidOptions( options );

	const resolutions = computeGridLevels( baseResolution, requestedLevels, mipsPerLevel );
	const levels = resolutions.length;
	const grids = resolutions.map( ( resolution ) => createLatentGrid( resolution, resolution, channels, random ) );

	const resolvedTextureResolution = textureResolution || resolutions[ 0 ];
	const lodOffset = computeFeatureLodOffset(resolvedTextureResolution,resolutions[0]);
	const maxLod = Math.floor( Math.log2( Math.max( 1, resolvedTextureResolution ) ) );

	const inputSize = computeDecoderInputSize( channels, positionalEncoding, dualGrid, lowResChannels );
	const decoder = createMLP( inputSize, hiddenSizes, outputChannels, random, hiddenActivation, 'linear' );

	const lowResGrids = dualGrid ? resolutions.map(r => createLatentGrid(Math.max(1, Math.floor(r/2)), Math.max(1, Math.floor(r/2)), lowResChannels, random)) : [];

	return { channels, lowResChannels, lowResGrids, levels, mipsPerLevel, resolutions, grids, decoder, hiddenSizes, hiddenActivation, outputChannels, textureResolution: resolvedTextureResolution, maxLod, lodOffset, positionalEncodingPeriod: options.positionalEncodingPeriod ?? 0, uvTransform, positionalEncoding, dualGrid };

}

export {
	createNTCGridPyramidModel,
	resolveNTCGridPyramidOptions,
	computeDecoderInputSize,
	createLatentGrid,
	computeGridLevels,
	LATENT_INIT_SCALE
};
export type { NTCGridPyramidModel, NTCGridPyramidOptions, ResolvedNTCGridPyramidOptions };
