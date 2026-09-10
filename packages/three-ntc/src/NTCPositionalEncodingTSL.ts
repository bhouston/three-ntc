import { abs, fract } from 'three/tsl';

// Optional "tiled positional encoding" from the NVIDIA neural texture
// compression paper (Section 4.3.2): 3 octaves (log2(8), 8 being the paper's
// max upsampling factor) x 2 phases x 2 axes (horizontal/vertical) = 12
// extra scalar decoder inputs, built from cheap triangle waves rather than
// sin/cos ("a more computationally efficient variant ... based on triangular
// waves ... observe no quality loss"). Shared verbatim by the GPU training
// kernel (NTCGPUComputeTSL.js, three-ntc-trainer) and the runtime decoder
// (NTCDecoderTSL.js) so the two can never disagree about this encoding.
//
// This addon applies it to the sub-texel fractional offset (tx, ty) within
// whichever single stored grid level a given LOD selects (see
// NTCMipBands.js) - the paper instead splits that offset across two grids
// (a "learned interpolation" high-res G0 + bilinear low-res G1); here the
// encoding pairs with the 4-neighbor-tap "learned interpolation" of the one
// selected level (see NTCDecoderTSL.js/NTCGPUComputeTSL.js's
// `positionalEncoding` branch). The optional `dualGrid` G1 tap (see
// NTCGridPyramidModel.js's `computeDecoderInputSize`) is plain bilinear and
// contributes nothing to this encoding, matching the paper.
const POSITIONAL_ENCODING_OCTAVES = 3;
const POSITIONAL_ENCODING_SIZE = POSITIONAL_ENCODING_OCTAVES * 2 * 2; // 12

/**
 * A period-1 triangle wave in [-1, 1], peaking at integer `x` - the paper's
 * cheap stand-in for `cos(2*pi*x)` (same shape, sharp corners instead of a
 * smooth curve). `triangleWaveTSL(x.sub(0.25))` gives the `sin(2*pi*x)`-like
 * quarter-period-shifted twin (zero and rising at integer `x`).
 */
function triangleWaveTSL( x: any ): any {

	return abs( fract( x ).sub( 0.5 ) ).mul( 4 ).sub( 1 );

}

/**
 * Builds the 12 positional-encoding scalars for one sub-texel offset
 * `(tx, ty)` (each typically in [0, 1), the fractional position within a
 * grid cell - see the 4-tap loops in NTCDecoderTSL.js/NTCGPUComputeTSL.js
 * that already compute this for bilinear/learned interpolation). Ordered
 * `[tx octave0 sin, tx octave0 cos, tx octave1 sin, tx octave1 cos, tx
 * octave2 sin, tx octave2 cos, ty ...(same 6)]`.
 */
function computeTiledPositionalEncodingTSL( tx: any, ty: any ): any[] {

	const values: any[] = [];

	for ( const t of [ tx, ty ] ) {

		for ( let h = 0; h < POSITIONAL_ENCODING_OCTAVES; h ++ ) {

			const freq = Math.pow( 2, h );
			values.push( triangleWaveTSL( t.mul( freq ).sub( 0.25 ) ) );
			values.push( triangleWaveTSL( t.mul( freq ) ) );

		}

	}

	return values;

}

export { computeTiledPositionalEncodingTSL, triangleWaveTSL, POSITIONAL_ENCODING_OCTAVES, POSITIONAL_ENCODING_SIZE };
