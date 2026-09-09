import { floor, int } from 'three/tsl';

// Maps a continuous/discrete target LOD (mip index) down onto one of the
// (far fewer) stored feature-level grids - shared by the training compute
// kernel (NTCGPUComputeTSL.js) and the runtime decoder (NTCDecoderTSL.js) so
// the two can never disagree about which grid level a given LOD selects.
// Kept in this parent (non-`training/`) folder, like NTCFormat.js, since the
// runtime decode path needs it without pulling in any training-only
// dependency.
//
// This is what reproduces the NVIDIA neural texture compression paper's
// actual storage-compression mechanism (Section 4.1/Table 1): a handful of
// stored feature levels (`levels`, small - e.g. 3 or 4), each reused, via
// the decoder's own explicit LOD input, to reconstruct `mipsPerLevel`
// different physical mip levels - not one stored grid per mip level (which
// is what a real GPU mipmap chain would be, and is *not* what this addon's
// pyramid is; see NTCGridModel.js's `computeGridLevels`). The mapping itself
// is intentionally uniform-width (`floor(lod / mipsPerLevel)`), not a
// faithful copy of Table 1's asymmetric per-level widths - see
// NTCGridModel.js's `DEFAULT_MIPS_PER_LEVEL` doc comment - but it keeps the
// paper's key property that the *last* stored level absorbs every LOD past
// its own band as an open-ended tail: once `lod` grows large enough that
// `floor(lod / mipsPerLevel) >= levels - 1`, every further LOD still clamps
// to that same last level, exactly like the paper's own last feature level
// "cannot be further downsampled".

/** Plain-JS version - used by tests and any CPU-side bookkeeping. */
function selectFeatureLevel( lod: number, levels: number, mipsPerLevel: number ): number {

	return Math.min( levels - 1, Math.max( 0, Math.floor( lod / mipsPerLevel ) ) );

}

/**
 * TSL node-graph equivalent of `selectFeatureLevel`, returning an int node -
 * `levels`/`mipsPerLevel` are plain JS numbers (baked into the generated
 * shader as literals, since they're fixed once a model's shape is known),
 * `lodNode` is the runtime LOD (a TSL float node).
 */
function selectFeatureLevelTSL( lodNode: any, levels: number, mipsPerLevel: number ): any {

	return int( floor( lodNode.div( mipsPerLevel ) ) ).clamp( 0, levels - 1 );

}

export { selectFeatureLevel, selectFeatureLevelTSL };
