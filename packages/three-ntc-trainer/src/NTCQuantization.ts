import { float, min, max, round } from 'three/tsl';

// Quantization-Aware Training (QAT) scheme registry, shared by every
// neural-* trainer (texture, material, appearance). Because every trainer's
// gradients are hand-derived (not autodiff - see NeuralGPUComputeTSL.js/
// NeuralTextureGPUComputeTSL.js/NeuralAppearanceGPUComputeTSL.js) and
// already treat a sampled latent as flowing through identity to the loss,
// a Straight-Through Estimator only needs a *forward* quantize step
// inserted where latents are sampled for MLP input - see the call sites in
// NeuralGridModel.js/NeuralTextureGPUComputeTSL.js/
// NeuralAppearanceGPUComputeTSL.js. The backward/gradient-accumulation
// kernels are untouched.
//
// Each scheme has the same shape - `quantizeForwardCPU(x, min, max)` (a
// plain JS function used by the CPU reference model/tests) and
// `quantizeForwardTSL(xNode, minNode, maxNode)` (the GPU-side mirror, built
// from the same 'three/tsl' node functions the rest of this codebase uses -
// see NeuralGPUComputeTSL.js/NeuralMLPTSL.js) - so adding a new scheme later
// is just adding another entry with both functions.
/**
 * Uniform `levels`-step scheme mirroring the matching `LATENT_CODECS` entry
 * in NTCBinaryCodec.ts (clamp to [min, max], round to the nearest level,
 * decode back) - the exact "simulated quantization" a Straight-Through
 * Estimator forward pass needs.
 */
function createUniformScheme( bits: number ) {

	const maxLevel = ( 1 << bits ) - 1;

	return {
		quantizeForwardCPU: ( x: number, lo = 0, hi = 1 ) => {

			const range = hi - lo;
			const t = range !== 0 ? Math.min( 1, Math.max( 0, ( x - lo ) / range ) ) : 0;

			return lo + ( Math.round( t * maxLevel ) / maxLevel ) * range;

		},
		quantizeForwardTSL: ( xNode: any, minNode: any, maxNode: any ) => {

			const range = maxNode.sub( minNode );
			const t = min( float( 1.0 ), max( float( 0.0 ), xNode.sub( minNode ).div( range ) ) );

			return minNode.add( round( t.mul( float( maxLevel ) ) ).div( float( maxLevel ) ).mul( range ) );

		}
	};

}

const QUANTIZATION_SCHEMES: Record<string, {
	quantizeForwardCPU: ( x: number, lo?: number, hi?: number ) => number;
	quantizeForwardTSL: ( xNode: any, minNode?: any, maxNode?: any ) => any;
}> = {
	none: {
		quantizeForwardCPU: ( x ) => x,
		quantizeForwardTSL: ( xNode ) => xNode
	},
	uint8: createUniformScheme( 8 ),
	uint4: createUniformScheme( 4 ),
	uint2: createUniformScheme( 2 )
};

const VALID_MODES = Object.keys( QUANTIZATION_SCHEMES );
const VALID_TARGETS = [ 'latents', 'weights', 'both' ];

interface NTCQuantizationOptions {
	mode?: string;
	target?: string;
	range?: 'auto' | [ number, number ];
	perLevel?: boolean;
}

interface ResolvedNTCQuantizationConfig {
	mode: string;
	target: string;
	range: 'auto' | [ number, number ];
	perLevel: boolean;
}

const DEFAULT_QUANTIZATION_OPTIONS: ResolvedNTCQuantizationConfig = {
	mode: 'none',
	target: 'latents',
	range: 'auto',
	perLevel: true
};

/**
 * Validates and defaults a trainer's `quantization` option, mirroring the
 * `NUMERIC_SETTINGS_SCHEMA`/`validateTrainingSettings` validation style in
 * NeuralAppearanceTrainer.js (clear thrown errors naming the bad field).
 * `target` currently only supports `'latents'` - `'weights'`/`'both'` are
 * accepted (reserved for a future weight-quantization scheme) but throw a
 * clear "not yet implemented" if actually selected, rather than silently
 * doing nothing.
 */
function resolveQuantizationConfig( options: { quantization?: NTCQuantizationOptions } = {} ): ResolvedNTCQuantizationConfig {

	const input = options.quantization || {};
	const mode = input.mode !== undefined ? input.mode : DEFAULT_QUANTIZATION_OPTIONS.mode;
	const target = input.target !== undefined ? input.target : DEFAULT_QUANTIZATION_OPTIONS.target;
	const range = input.range !== undefined ? input.range : DEFAULT_QUANTIZATION_OPTIONS.range;
	const perLevel = input.perLevel !== undefined ? input.perLevel : DEFAULT_QUANTIZATION_OPTIONS.perLevel;

	if ( QUANTIZATION_SCHEMES[ mode ] === undefined ) {

		throw new Error( `THREE.NTCQuantization: quantization.mode must be one of [${ VALID_MODES.join( ', ' ) }], got "${ mode }".` );

	}

	if ( VALID_TARGETS.includes( target ) === false ) {

		throw new Error( `THREE.NTCQuantization: quantization.target must be one of [${ VALID_TARGETS.join( ', ' ) }], got "${ target }".` );

	}

	if ( target !== 'latents' && mode !== 'none' ) {

		throw new Error( `THREE.NTCQuantization: quantization.target "${ target }" is not yet implemented - only "latents" is currently supported.` );

	}

	if ( range !== 'auto' ) {

		const isRangeTuple = Array.isArray( range ) && range.length === 2 &&
			Number.isFinite( range[ 0 ] ) && Number.isFinite( range[ 1 ] ) && range[ 0 ] <= range[ 1 ];

		if ( isRangeTuple === false ) {

			throw new Error( 'THREE.NTCQuantization: quantization.range must be "auto" or a [min, max] tuple with min <= max.' );

		}

	}

	if ( typeof perLevel !== 'boolean' ) {

		throw new Error( 'THREE.NTCQuantization: quantization.perLevel must be a boolean.' );

	}

	return { mode, target, range, perLevel };

}

interface GridLevelLayout {
	offset: number;
	floatCount: number;
	[key: string]: unknown;
}

/**
 * CPU-side per-level (or global) min/max reduction over a flat latent-grid
 * Float32Array, used to implement `quantization.range === 'auto'` (see
 * `resolveQuantizationConfig` above). `gridLevels` is the `layout.gridLevels`
 * array every trainer's GPUModel already builds (`{ offset, floatCount, ... }`
 * - see NTCGPUModel.js/(removed, appearance deleted)'s
 * `computeXXXModelLayout`), so this works identically for both trainers
 * without either one re-deriving level byte ranges. Grids are small (a few
 * thousand texels at most per level), so a plain CPU scan is fast enough -
 * no GPU reduction kernel is needed (see NeuralAppearanceTrainer.js/
 * NTCTrainer.js for how often this actually runs).
 *
 * Returns one `[min, max]` tuple per grid level. When `perLevel` is false,
 * every level gets the *same* tuple - the global min/max across every level
 * - rather than the caller having to special-case a single shared range
 * downstream (the GPU kernels always index one range per level regardless of
 * `perLevel`). A level with zero texels (degenerate, shouldn't normally
 * happen) or a perfectly flat buffer falls back to `[-1, 1]` rather than
 * `[Infinity, -Infinity]`/`[x, x]`, mirroring `encodeUint8Base64`'s
 * `min === max` no-NaN guard.
 */
function computeLatentRanges( data: Float32Array, gridLevels: GridLevelLayout[], perLevel = true ): Array<[ number, number ]> {

	const levelRanges: Array<[ number, number ]> = gridLevels.map( ( level ) => {

		let lo = Infinity;
		let hi = - Infinity;

		for ( let i = level.offset; i < level.offset + level.floatCount; i ++ ) {

			const value = data[ i ];
			if ( value < lo ) lo = value;
			if ( value > hi ) hi = value;

		}

		if ( lo > hi ) return [ - 1, 1 ];

		return [ lo, hi ];

	} );

	if ( perLevel ) return levelRanges;

	let lo = Infinity;
	let hi = - Infinity;

	for ( const [ levelLo, levelHi ] of levelRanges ) {

		if ( levelLo < lo ) lo = levelLo;
		if ( levelHi > hi ) hi = levelHi;

	}

	const globalRange: [ number, number ] = lo > hi ? [ - 1, 1 ] : [ lo, hi ];

	return gridLevels.map( () => globalRange );

}

/**
 * Reads the current latent-grid buffer back from the GPU, recomputes the
 * per-level `range: 'auto'` min/max (see `computeLatentRanges`), and pushes
 * it into `gpuModel`'s quantization-range uniforms (see
 * `setQuantizationRange` on NTCGPUModel.js/
 * (removed, appearance deleted)) - shared by NTCTrainer.js/
 * NeuralAppearanceTrainer.js so both call the exact same
 * readback-then-rescan sequence. A no-op (skips the readback entirely) when
 * quantization is disabled or the range is a fixed tuple - only `'auto'`
 * ever needs re-measuring.
 */
async function refreshGPUQuantizationRange( gpuModel: any, renderer: any ): Promise<void> {

	if ( gpuModel.quantization.mode === 'none' || gpuModel.quantization.range !== 'auto' ) return;

	const buffer = await renderer.getArrayBufferAsync( gpuModel.latentsBuffers.attribute );
	const latents = new Float32Array( buffer );
	const ranges = computeLatentRanges( latents, gpuModel.layout.gridLevels, gpuModel.quantization.perLevel );

	gpuModel.setQuantizationRange( ranges );

}

export {
	QUANTIZATION_SCHEMES,
	DEFAULT_QUANTIZATION_OPTIONS,
	resolveQuantizationConfig,
	computeLatentRanges,
	refreshGPUQuantizationRange
};
export type { NTCQuantizationOptions, ResolvedNTCQuantizationConfig, GridLevelLayout };
