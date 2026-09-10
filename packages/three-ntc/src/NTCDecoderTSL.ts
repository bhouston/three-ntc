import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	createMat4Storage,
	createVec4Storage
} from './NTCMLPTSL.js';
import { buildMipChainTexture, buildLevelTextures, NTCGrid } from './NTCHalfFloatTexture.js';
import { selectFeatureLevelTSL } from './NTCMipBands.js';
import { computeTiledPositionalEncodingTSL } from './NTCPositionalEncodingTSL.js';
import { MLPLayer } from './NTCBinaryCodec.js';
import { float, floor, int, textureLevel, vec2 } from 'three/tsl';

/**
 * A fully decoded `.ntc` CPU model - what `NTCLoader.parse()` produces and
 * `NTCNodeMaterial`'s constructor / `evaluateNeuralTextureRaw` consume.
 */
export interface NTCCpuModel {
	channels: number;
	levels: number;
	mipsPerLevel: number;
	maxLod: number;
	grids: NTCGrid[];
	decoder: { layers: MLPLayer[] };
	outputChannels: number;
	wrap: string;
	uvTransform: any;
	// Optional (default false/absent) - see NTCGridPyramidModel.js (trainer)
	// `computeDecoderInputSize` doc comment and `evaluateNeuralTextureRaw`
	// below.
	positionalEncoding?: boolean;
	// Optional (default false/absent) - the paper's G0/G1 pair, see
	// `computeDecoderInputSize` and `sampleCoarsestLevelBilinear` below.
	dualGrid?: boolean;
}

/**
 * Builds the `positionalEncoding` decoder input (NVIDIA neural texture
 * compression paper Section 4.3, adapted to this addon's single-selected-
 * level design - see NTCGridPyramidModel.js's `computeDecoderInputSize` doc
 * comment): 4 raw neighbor taps ("learned interpolation") concatenated from
 * whichever single stored grid level `lodNode` selects, plus 12 tiled
 * positional-encoding scalars built from that same level's own sub-texel
 * offset - mirrors NTCGPUComputeTSL.js's training kernel's forward pass
 * exactly (same level-selection, same tap coordinates, same encoding), which
 * is required since the decoder MLP was fit against that exact input
 * distribution.
 *
 * Deliberately hard-selects one level (`selectFeatureLevelTSL`, matching
 * training bit-for-bit) rather than reusing `buildMipChainTexture`'s smooth
 * hardware-trilinear blend *between* levels (see this file's default,
 * non-`positionalEncoding` path, and NTCHalfFloatTexture.js's doc comment
 * on why that blend exists there): a raw 4-tap fetch has no filtered
 * "in-between" representation to blend, so a fractional LOD straddling two
 * stored levels' bands snaps discontinuously here instead of cross-fading -
 * a real (if less-visible-in-practice, since `mipsPerLevel` already spreads
 * one stored level across more than one physical mip) simplification versus
 * the default path.
 */
function evaluatePositionalEncodingFeatures( uvNode: any, cpuModel: NTCCpuModel, levelTextures: any[], lodNode: any ): any[] {

	const channels = cpuModel.channels;
	const grids = cpuModel.grids;
	const selectedLevel = selectFeatureLevelTSL( lodNode, grids.length, cpuModel.mipsPerLevel );

	// Accumulated as plain expressions (`a = a.add(...)`), not `.toVar()` +
	// `.addAssign()`: this runs in a material's node graph, outside any
	// `Fn()` scope, where TSL has no statement stack and silently drops
	// assignments ("No stack defined for assign operation") - which left
	// every tap and the sub-texel offset at 0, so a positionalEncoding
	// model rendered garbage while its training kernel (inside a Fn) was
	// fine. Caught by NTCDecoder.gpu.test.ts.
	const taps: any[] = [];
	for ( let t = 0; t < 4 * channels; t ++ ) taps.push( float( 0 ) );
	let selTx: any = float( 0 );
	let selTy: any = float( 0 );

	for ( let g = 0; g < grids.length; g ++ ) {

		const grid = grids[ g ];
		const width = grid.width;
		const height = grid.height;

		const x = uvNode.x.mul( width ).sub( 0.5 );
		const y = uvNode.y.mul( height ).sub( 0.5 );
		const x0 = floor( x );
		const y0 = floor( y );
		const tx = x.sub( x0 );
		const ty = y.sub( y0 );

		const weight = selectedLevel.equal( int( g ) ).select( float( 1 ), float( 0 ) );

		// Texel-center UVs of the 4 neighboring texels - RepeatWrapping on
		// `levelTextures[g]` (see buildLevelTextures) handles wraparound at
		// the edges for free, matching NTCGPUComputeTSL.js's `wrapIndexTSL`
		// without reproducing its manual index math here.
		const corners = [
			[ x0.add( 0.5 ).div( width ), y0.add( 0.5 ).div( height ) ],
			[ x0.add( 1.5 ).div( width ), y0.add( 0.5 ).div( height ) ],
			[ x0.add( 0.5 ).div( width ), y0.add( 1.5 ).div( height ) ],
			[ x0.add( 1.5 ).div( width ), y0.add( 1.5 ).div( height ) ]
		];

		for ( let t = 0; t < 4; t ++ ) {

			const sample = textureLevel( levelTextures[ g ], vec2( corners[ t ][ 0 ], corners[ t ][ 1 ] ), 0 );
			const comps = [ sample.x, sample.y, sample.z, sample.w ];

			for ( let c = 0; c < channels; c ++ ) taps[ t * channels + c ] = taps[ t * channels + c ].add( comps[ c ].mul( weight ) );

		}

		selTx = selTx.add( tx.mul( weight ) );
		selTy = selTy.add( ty.mul( weight ) );

	}

	const pe = computeTiledPositionalEncodingTSL( selTx, selTy );

	return [ ...taps, ...pe ];

}

/**
 * The `dualGrid` G1 tap (see NTCGridPyramidModel.js's
 * `computeDecoderInputSize`): one plain bilinear, LOD-independent sample of
 * the *coarsest* stored level. Reads the mip-chain texture at that level's
 * own native physical mip (band start, see NTCHalfFloatTexture.js's
 * `buildMipChainLevels`) when available, else - the `positionalEncoding`
 * path, which has only raw `NearestFilter` per-level textures - blends the
 * 4 neighbor texels manually, exactly as NTCGPUComputeTSL.js's training
 * kernel does.
 */
function sampleCoarsestLevelBilinear( uvNode: any, cpuModel: NTCCpuModel, mipChainTexture: any, levelTextures: any[] | null ): any[] {

	const channels = cpuModel.channels;
	const last = cpuModel.grids.length - 1;

	if ( mipChainTexture ) {

		const sample = textureLevel( mipChainTexture, uvNode, float( last * cpuModel.mipsPerLevel ) );
		return [ sample.x, sample.y, sample.z, sample.w ].slice( 0, channels );

	}

	const { width, height } = cpuModel.grids[ last ];
	const x = uvNode.x.mul( width ).sub( 0.5 );
	const y = uvNode.y.mul( height ).sub( 0.5 );
	const x0 = floor( x );
	const y0 = floor( y );
	const tx = x.sub( x0 );
	const ty = y.sub( y0 );

	const tap = ( dx: number, dy: number ) => textureLevel( levelTextures![ last ], vec2( x0.add( dx + 0.5 ).div( width ), y0.add( dy + 0.5 ).div( height ) ), 0 );
	const s00 = tap( 0, 0 ), s10 = tap( 1, 0 ), s01 = tap( 0, 1 ), s11 = tap( 1, 1 );
	const blended = s00.mul( float( 1 ).sub( tx ).mul( float( 1 ).sub( ty ) ) )
		.add( s10.mul( tx.mul( float( 1 ).sub( ty ) ) ) )
		.add( s01.mul( float( 1 ).sub( tx ).mul( ty ) ) )
		.add( s11.mul( tx.mul( ty ) ) );

	return [ blended.x, blended.y, blended.z, blended.w ].slice( 0, channels );

}

/**
 * Builds the TSL expression that evaluates the trained mip pyramid + MLP
 * decoder at `uvNode`, returning the raw array of `outputChannels` scalar
 * nodes (one per trained channel - callers slice/decode these into whatever
 * physical quantities they represent, see NTCFormat.js).
 * `lodNode` is the requested LOD (mip index, a TSL float node - e.g. derived
 * from screen-space UV derivatives or an explicit distance-based estimate,
 * see NTCNodeMaterial.js) this decode should reconstruct - defaults to
 * `float(0)` (finest/closest LOD) when omitted. It drives one single
 * hardware `textureSampleLevel` call against `mipChainTexture` (built by
 * NTCHalfFloatTexture.js's `buildMipChainTexture`): the GPU brackets
 * `lodNode` between its two nearest physical mip levels and blends them
 * (genuine trilinear - bilinear within each mip, linear between the two),
 * so a fractional LOD - including one that straddles two *different* stored
 * feature levels' bands - reconstructs a smooth cross-fade instead of the
 * old hard 0/1 level-equality switch. The normalized LOD is still
 * concatenated onto the decoder's input exactly as before - this must match
 * training bit-for-bit, or the decoder sees an input distribution it was
 * never fit against.
 *
 * `levelTextures` (see NTCHalfFloatTexture.buildLevelTextures) is only
 * needed - and only consulted - when `cpuModel.positionalEncoding` is true;
 * `mipChainTexture` is ignored in that case (see
 * `evaluatePositionalEncodingFeatures`'s doc comment for why that path
 * can't reuse the smooth cross-level mip-chain trick).
 */
function evaluateNeuralTextureRaw( uvNode: any, cpuModel: NTCCpuModel, mipChainTexture: any, renderer: any | null = null, lodNode: any = null, levelTextures: any[] | null = null ): any[] {

	// renderer is accepted for compatibility with the higher-level material
	// constructor, but this stock-Three.js path always uses fp32 uniforms.
	void renderer;

	const resolvedLodNode = lodNode || float( 0 );
	const channels = cpuModel.channels;

	let features: any[];

	if ( cpuModel.positionalEncoding ) {

		if ( ! levelTextures ) {

			throw new Error( 'THREE.NTCDecoderTSL: a positionalEncoding model requires levelTextures (see NTCHalfFloatTexture.buildLevelTextures).' );

		}

		features = evaluatePositionalEncodingFeatures( uvNode, cpuModel, levelTextures, resolvedLodNode );

	} else {

		const sample = textureLevel( mipChainTexture, uvNode, resolvedLodNode );
		features = [ sample.x, sample.y, sample.z, sample.w ].slice( 0, channels );

	}

	// G1 (dualGrid) - concatenated after G0's taps, before the LOD scalar,
	// matching NTCGPUComputeTSL.js's input layout exactly.
	if ( cpuModel.dualGrid ) features.push( ...sampleCoarsestLevelBilinear( uvNode, cpuModel, mipChainTexture, levelTextures ) );

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `computeDecoderInputSize` /
	// NTCGPUComputeTSL.js's forward pass exactly.
	// Math.max(1, ...) guards against a genuine maxLod of 0 (a model that
	// only ever supports LOD 0, see NTCGridPyramidModel.js) - dividing by 0
	// there would produce a NaN feature, matching the same guard
	// NTCGPUComputeTSL.js's training kernel already applies.
	features.push( resolvedLodNode.div( Math.max( 1, cpuModel.maxLod ) ) );

	// Shared mat4-packed MLP evaluator (see NTCMLPTSL.js). Packing weights
	// into 4x4 blocks and evaluating each layer with a native mat4 * vec4
	// multiply maps to one hardware FMA-chain instruction per input quad
	// (instead of 4 separate dot() calls, one per output neuron), and
	// evaluateLinearLayerMat4 materializes each layer's output with .toVar()
	// before the next layer consumes it - see that function's doc comment
	// for the "maximum parser recursive depth" WGSL failure this works
	// around.
	let activations = packVec4Inputs( features );

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const weights = createMat4Storage( packLayerWeightsMat4( layer.weights, layer.inputSize, layer.outputSize ) );
		const biases = createVec4Storage( packLayerBiasesVec4( layer.biases ) );
		const inputVectorCount = Math.ceil( layer.inputSize / 4 );

		activations = evaluateLinearLayerMat4(
			activations, layer.inputSize, layer.outputSize, layer.activation,
			( outputVector: number, inputVector: number ) => weights.node.element( outputVector * inputVectorCount + inputVector ),
			( outputVector: number ) => biases.node.element( outputVector )
		);

	}

	const lastLayer = cpuModel.decoder.layers[ cpuModel.decoder.layers.length - 1 ];

	return unpackVec4Outputs( activations, lastLayer.outputSize );

}

export { evaluateNeuralTextureRaw, buildMipChainTexture, buildLevelTextures };
