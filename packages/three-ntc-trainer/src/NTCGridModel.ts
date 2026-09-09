const LATENT_INIT_SCALE = 0.35;

// Single source of truth for the acceptable/offered values of the grid + MLP
// shape options (grid feature levels, grid finest resolution, MLP hidden
// width) - shared by every example's GUI dropdown so their offered ranges
// can't silently drift apart. These are display/GUI option lists, not hard
// validation: `computeGridLevels` itself accepts any positive
// `baseResolution`/`levels`/`mipsPerLevel` combination, not just these.
const GRID_LEVELS_OPTIONS = [ 1, 2, 3, 4, 5, 6, 8 ];
const GRID_BASE_RESOLUTION_OPTIONS = [ 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 ];
const MLP_HIDDEN_SIZE_OPTIONS = [ 8, 16, 32, 64, 128 ];
// Hidden-layer activations offered in the GUI (see
// NTCGridPyramidModel.js's `hiddenActivation` option / NTCMLP.js's
// hardGELU doc comment) - 'relu' is the cheapest (a single max() per
// neuron, see NTCMLPTSL.js) and remains the default; 'hgelu' is the NVIDIA
// neural texture compression paper's own hidden activation (Section 4.4), a
// 3-piece GELU approximation, usually a quality win at a small extra ALU
// cost per neuron.
const MLP_ACTIVATION_OPTIONS = [ 'relu', 'hgelu' ];

// How many mip levels of an actual mipmap chain each stored feature level
// covers by default (see this file's module doc comment, and
// NTCMipBands.js's `selectFeatureLevel`) - the NVIDIA neural texture
// compression paper's own Table 1 uses 2 for every level but its first
// (which is wider, since it has no finer neighbor to share the load with)
// and its last (an open-ended tail). We use a uniform width for every level
// including the first - a deliberate simplification, not a faithful copy of
// Table 1's asymmetric widths - while still keeping the paper's core
// mechanism: a level's grid is reused, via the decoder's explicit LOD input,
// to reconstruct more than one physical mip level, and the *last* stored
// level absorbs every mip past its own band as a tail (see
// `selectFeatureLevel`), matching "the last feature level ... cannot be
// further downsampled".
const DEFAULT_MIPS_PER_LEVEL = 2;

// Hard ceiling on the finest (base) grid level's resolution - independent of
// the GUI option list above, this is enforced by `computeGridLevels` itself
// so no caller (GUI-driven or programmatic) can ever request a latent grid
// texture large enough to be impractical (a 4096x4096 level alone is already
// 64x the texel count of a 512x512 one). Every subsequent (coarser) level's
// resolution only ever shrinks from the previous one's, so it's only the
// finest level that needs this ceiling.
const MAX_GRID_RESOLUTION = 4096;

/**
 * Computes the per-level grid resolutions for a genuine (paper-style) mip
 * pyramid: `baseResolution` is the finest (mip 0) level's resolution, and
 * each subsequent stored *feature* level is `mipsPerLevel` mip-halvings
 * coarser than the previous one - e.g. `computeGridLevels(128, 3, 2)` ->
 * `[128, 32, 8]` (each level a quarter the resolution of the last, i.e. 2
 * halvings apart). This is deliberately not a 1:1 grid-level-per-mip-level
 * mapping - see NTCMipBands.js's `selectFeatureLevel`, which maps a
 * continuous/discrete target LOD down onto one of these (far fewer) stored
 * levels, exactly like the NVIDIA neural texture compression paper's Table 1
 * (a handful of feature levels, each reused via the decoder's LOD input to
 * reconstruct several physical mips - see that file's doc comment for how
 * this reproduces the paper's actual storage-compression mechanism).
 *
 * Level 0 (index 0, the array's first entry) is always the finest/base
 * resolution. `baseResolution` is clamped to `[1, MAX_GRID_RESOLUTION]` and
 * rounded to the nearest integer; `levels` and `mipsPerLevel` are each
 * clamped to >= 1 and rounded.
 */
function computeGridLevels( baseResolution: number, levels: number, mipsPerLevel: number = DEFAULT_MIPS_PER_LEVEL ): number[] {

	const resolutions: number[] = [];
	const count = Math.max( 1, Math.round( levels ) );
	const step = Math.max( 1, Math.round( mipsPerLevel ) );
	let resolution = Math.min( MAX_GRID_RESOLUTION, Math.max( 1, Math.round( baseResolution ) ) );

	for ( let i = 0; i < count; i ++ ) {

		resolutions.push( resolution );

		for ( let k = 0; k < step; k ++ ) resolution = Math.max( 1, Math.floor( resolution / 2 ) );

	}

	return resolutions;

}

interface LatentGrid {
	width: number;
	height: number;
	channels: number;
	data: Float32Array;
}

function createLatentGrid( width: number, height: number, channels: number, random: () => number ): LatentGrid {

	const data = new Float32Array( width * height * channels );

	for ( let i = 0; i < data.length; i ++ ) {

		data[ i ] = ( random() * 2 - 1 ) * LATENT_INIT_SCALE;

	}

	return { width, height, channels, data };

}

export {
	computeGridLevels,
	createLatentGrid,
	LATENT_INIT_SCALE,
	GRID_LEVELS_OPTIONS,
	GRID_BASE_RESOLUTION_OPTIONS,
	MLP_HIDDEN_SIZE_OPTIONS,
	MLP_ACTIVATION_OPTIONS,
	MAX_GRID_RESOLUTION,
	DEFAULT_MIPS_PER_LEVEL
};
export type { LatentGrid };
