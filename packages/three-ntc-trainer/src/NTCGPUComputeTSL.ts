import { atomicAddFloat, atomicLoadFloat } from './NTCFloatAtomic.js';
import {
	Fn,
	If,
	Loop,
	atomicAdd,
	atomicLoad,
	float,
	floor,
	instanceIndex,
	int,
	textureLevel,
	textureSize,
	texture,
} from 'three/tsl';
import { trainingRandomTSL, trainingUVTSL, trainingLodTSL } from './NTCSampling.js';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from './NTCGPUTrainingConstants.js';
import {
	wrapIndexTSL,
	forwardDenseLayerTSL,
	accumulateDenseLayerGradTSL,
	backwardDenseLayerTSL,
	createAdamComputeNode,
	type TSLNode
} from './NTCGPUKernelsTSL.js';
import { applyChannelActivation, channelActivationDerivativeFromOutput } from 'three-ntc';
import { QUANTIZATION_SCHEMES } from './NTCQuantization.js';
import { selectFeatureLevelTSL, computeTiledPositionalEncodingTSL, POSITIONAL_ENCODING_SIZE } from 'three-ntc';
import type { NTCGPUModel } from './NTCGPUModel.js';

/**
 * Creates the training compute node: samples the source texture(s) directly
 * (no teacher-atlas readback needed, since the target is already a static
 * GPU texture), forward-evaluates the mip pyramid + MLP decoder, computes an
 * L2 loss, and hand-differentiates the backward pass - accumulating
 * gradients atomically exactly like the neural appearance trainer's GPU
 * backward pass. Runs one invocation per batch sample.
 *
 * Every sample trains against exactly one, stochastically chosen, integer
 * mip level (see `sampleTrainingLod`, sampled across the model's full
 * `maxLod` mip range - see NTCGridPyramidModel.js) - the source texture is
 * sampled at that exact mip, and that LOD is mapped down onto exactly one
 * *stored* grid level (`selectFeatureLevelTSL`, see NTCMipBands.js - the
 * pyramid stores far fewer levels than `maxLod`, each one reused to
 * reconstruct several physical mips, matching the NVIDIA neural texture
 * compression paper's Table 1 rather than a real per-mip GPU mip chain). The
 * decoder's input is that one selected grid level's `channels`-wide bilinear
 * tap plus the normalized LOD itself - the LOD is what lets the *same*
 * stored level's decode differ correctly across the several physical mips it
 * covers (see NTCGridPyramidModel.js's doc comment for why the decoder is
 * always `channels + 1` wide, independent of how many levels are stored).
 * No cross-level blending happens during training - only ever one level is
 * selected per sample - that only happens at inference, via interpolation
 * between two independently-trained neighboring levels (see
 * NTCDecoderTSL.js).
 *
 * `sourceTextures` is an array of RGBA textures whose channels are
 * concatenated (in order, up to 4 components each) to form the
 * `outputChannels`-wide training target - e.g. a single albedo texture for
 * the texture-fitting demo, or 5 packed textures for the full-material demo
 * (see NeuralMaterialFormat.js). Each must have a real mip chain
 * (`generateMipmaps: true` or a manually supplied `.mipmaps`) reaching at
 * least `maxLod` levels deep, since sampling stops each sample's chosen LOD
 * from *this* texture, not from the latent grids.
 *
 * `gpuModel.layout.channelActivations`, if present, is a flat array (one
 * entry per output channel, `undefined`/omitted entries default to plain
 * linear) naming an output nonlinearity applied to that channel's raw
 * decoder output before the loss/delta are computed - see
 * ./NTCOutputActivations.js. Used by neural-material to fit each
 * channel's targets in their own natural physical range (bounded reflectance
 * via sigmoid, signed tangent-space offsets via tanh, HDR emission via
 * softplus) instead of forcing every channel through the same unbounded
 * linear output.
 */
function createTextureTrainBatchComputeNode( gpuModel: NTCGPUModel, sourceTextures: TSLNode[], samples: { uv?: TSLNode; lod?: TSLNode; trainLatents?: boolean } = {} ): TSLNode {

	const {
		layout,
		batchSize,
		activationsStorage,
		lossAtomic,
		stepUniform,
		quantization,
		quantizationRangeUniforms
	} = gpuModel;
	const addGradient = (destination: any, value: any) => {
		if (gpuModel.floatGradients) atomicAddFloat(destination, value);
		else atomicAdd(destination, int(value.mul(FIXED_POINT_SCALE)));
	};
	const { valuesStorage: weightsStorage, gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { valuesStorage: latentsStorage, gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;
	// Quantization is applied to stored taps before interpolation. Noise QAT
	// uses fixed bounds; STE remains available with explicit or auto ranges.
	const quantizeLatent = quantization.mode !== 'none' ?
		QUANTIZATION_SCHEMES[ quantization.mode ].quantizeForwardTSL :
		null;

	const {
		gridLevels,
		channels,
		lowResChannels,
		mlpLayers,
		a0Offset,
		layerActs,
		deltaOffsets,
		gradA0Offset,
		activationStride,
		outputChannels,
		channelActivations,
		mipsPerLevel,
		maxLod,
		positionalEncoding,
		dualGrid
	} = layout;

	// See NTCGridPyramidModel.js's `computeDecoderInputSize` doc comment - the
	// width of the selected level's own tap(s), not counting the trailing LOD
	// scalar: a plain `channels`-wide bilinear tap, or (positionalEncoding)
	// `4 * channels` raw neighbor taps + `POSITIONAL_ENCODING_SIZE` (12) tiled
	// positional-encoding scalars.
	//
	// `dualGrid` appends one more `channels`-wide plain bilinear tap of the
	// matching half-resolution G1 grid after those.
	const g0Width = positionalEncoding ? channels * 4 + POSITIONAL_ENCODING_SIZE : channels;
	const featureWidth = g0Width + ( dualGrid ? lowResChannels : 0 );


	return Fn( () => {

		const sampleIdx = int( instanceIndex );
		const randomUV = samples.uv ?? trainingUVTSL( sampleIdx, stepUniform );
		const actBase = sampleIdx.mul( int( activationStride ) );

		// This sample's stochastically chosen, exact-integer training LOD,
		// across the model's full physical mip range - see
		// sampleTrainingLod's doc comment - and the *stored* grid level it
		// maps onto (see this function's doc comment and NTCMipBands.js).
		const lod = samples.lod ?? trainingLodTSL( sampleIdx, stepUniform, maxLod );
		// Train exact source texels; bilinear targets blur away high frequencies.
		const sourceSize = textureSize(texture(sourceTextures[0]), int(lod));
		const uv = samples.uv ?? floor(randomUV.mul(sourceSize)).add(0.5).div(sourceSize);
		const selectedLevel = selectFeatureLevelTSL( lod, layout.levels, mipsPerLevel, layout.lodOffset );

		const targetComponents: TSLNode[] = [];

		for ( const sourceTexture of sourceTextures ) {

			const sample = textureLevel( sourceTexture, uv, lod );
			const remaining = outputChannels - targetComponents.length;

			if ( remaining > 0 ) targetComponents.push( sample.x );
			if ( remaining > 1 ) targetComponents.push( sample.y );
			if ( remaining > 2 ) targetComponents.push( sample.z );
			if ( remaining > 3 ) targetComponents.push( sample.w );

		}

		// 1. Sample every grid level (wrap addressing), but keep only the one
		// matching `selectedLevel` - every other level's contribution is
		// multiplied by an exact 0/1 selector (`weight`) rather than being
		// omitted from the shader, since which level is selected is a
		// per-invocation runtime value, not something known at kernel-build
		// time (unlike `gridLevels.length` itself, which is why this loop is
		// still unrolled in JS). Accumulated into a single shared,
		// `featureWidth`-wide `a0Vars` (not a per-level slot - see
		// NTCGridPyramidModel.js's doc comment on why the decoder input no
		// longer scales with level count): since `weight` is 1 for exactly
		// one `g` and 0 for every other, the sum equals that one level's
		// value(s).
		//
		// Two modes (see `computeDecoderInputSize`'s doc comment):
		// `positionalEncoding === false` (default) bilinearly blends the 4
		// neighbor taps into one `channels`-wide value per level, matching
		// this addon's original simpler input. `positionalEncoding === true`
		// instead concatenates the 4 raw (unblended) taps - "learned
		// interpolation", NVIDIA paper Section 4.3.1 - and appends 12 tiled
		// positional-encoding scalars (Section 4.3.2) built from the
		// selected level's own sub-texel offset, which is what lets the MLP
		// recover the phase information a plain concatenation of unordered
		// taps would otherwise lose.
		const levelTaps: Array<{ off0: TSLNode; off1: TSLNode; off2: TSLNode; off3: TSLNode; w0: TSLNode; w1: TSLNode; w2: TSLNode; w3: TSLNode; weight: TSLNode }> = [];
		const a0Vars: TSLNode[] = [];
		for ( let c = 0; c < featureWidth; c ++ ) a0Vars.push( float( 0.0 ).toVar() );
		const selTx = positionalEncoding ? float( 0.0 ).toVar() : null;
		const selTy = positionalEncoding ? float( 0.0 ).toVar() : null;

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const level = gridLevels[ g ];
			const x = uv.x.mul( level.width ).sub( 0.5 );
			const y = uv.y.mul( level.height ).sub( 0.5 );
			const x0 = int( floor( x ) );
			const y0 = int( floor( y ) );
			const tx = x.sub( float( x0 ) );
			const ty = y.sub( float( y0 ) );

			const w0 = float( 1.0 ).sub( tx ).mul( float( 1.0 ).sub( ty ) );
			const w1 = tx.mul( float( 1.0 ).sub( ty ) );
			const w2 = float( 1.0 ).sub( tx ).mul( ty );
			const w3 = tx.mul( ty );

			const tapX0 = wrapIndexTSL( x0, level.width );
			const tapY0 = wrapIndexTSL( y0, level.height );
			const tapX1 = wrapIndexTSL( x0.add( 1 ), level.width );
			const tapY1 = wrapIndexTSL( y0, level.height );
			const tapX2 = wrapIndexTSL( x0, level.width );
			const tapY2 = wrapIndexTSL( y0.add( 1 ), level.height );
			const tapX3 = wrapIndexTSL( x0.add( 1 ), level.width );
			const tapY3 = wrapIndexTSL( y0.add( 1 ), level.height );

			const off0 = int( level.offset ).add( tapY0.mul( level.width ).add( tapX0 ).mul( level.channels ) );
			const off1 = int( level.offset ).add( tapY1.mul( level.width ).add( tapX1 ).mul( level.channels ) );
			const off2 = int( level.offset ).add( tapY2.mul( level.width ).add( tapX2 ).mul( level.channels ) );
			const off3 = int( level.offset ).add( tapY3.mul( level.width ).add( tapX3 ).mul( level.channels ) );

			const weight = selectedLevel.equal( int( level.featureLevel ) ).select( float( 1 ), float( 0 ) );

			levelTaps.push( { off0, off1, off2, off3, w0, w1, w2, w3, weight } );

			// Quantization belongs to stored values, before any interpolation.
			const readTap = ( offset: TSLNode, c: number ): TSLNode => {
				const value = latentsStorage.element( offset.add( c ) );
				if (!quantizeLatent) return value;
				const {min:lo,max:hi}=quantizationRangeUniforms[g];
				if (quantization.method === 'noise') {
					const bins=2 ** Number(quantization.mode.slice(4));
					return value.add(trainingRandomTSL(offset.add(c),stepUniform,11).sub(0.5)
						.mul(hi.sub(lo).div(bins-1)).mul(gpuModel.quantizationNoiseUniform));
				}
				return quantizeLatent(value,lo,hi);
			};
			const readBilinear = ( c: number ): TSLNode => readTap(off0,c).mul(w0)
				.add(readTap(off1,c).mul(w1)).add(readTap(off2,c).mul(w2)).add(readTap(off3,c).mul(w3));

			if ( level.isLowRes ) {
				for (let c=0;c<lowResChannels;c++) a0Vars[g0Width+c].addAssign(readBilinear(c).mul(weight));
				continue;
			}

			if ( positionalEncoding ) {

				const offs = [ off0, off1, off2, off3 ];

				for ( let t = 0; t < 4; t ++ ) {

					for ( let c = 0; c < channels; c ++ ) {

						const raw_c = latentsStorage.element( offs[ t ].add( c ) );
						// QAT forward quantize (STE), same as the bilinear path
						// below - see its comment.
						const quantized_c = quantizeLatent !== null ?
							quantizeLatent( raw_c, quantizationRangeUniforms[ g ].min, quantizationRangeUniforms[ g ].max ) :
							raw_c;

						a0Vars[ t * channels + c ].addAssign( quantized_c.mul( weight ) );

					}

				}

				selTx!.addAssign( tx.mul( weight ) );
				selTy!.addAssign( ty.mul( weight ) );

			} else {

				for ( let c = 0; c < channels; c ++ ) a0Vars[ c ].addAssign( readBilinear( c ).mul( weight ) );

			}

		}

		// Tiled positional encoding (see NTCPositionalEncodingTSL.js) of the
		// selected level's own sub-texel offset - computed once after the
		// g-loop above resolves `selTx`/`selTy` to that one level's `(tx, ty)`
		// (every other level contributed 0, same one-hot-`weight` trick as
		// the taps themselves).
		if ( positionalEncoding ) {

			const phase = uv.mul(sourceSize).div(layout.positionalEncodingPeriod || 1);
			const pe = computeTiledPositionalEncodingTSL(
				layout.positionalEncodingPeriod ? phase.x : selTx!,
				layout.positionalEncodingPeriod ? phase.y : selTy! );
			for ( let k = 0; k < pe.length; k ++ ) a0Vars[ channels * 4 + k ].assign( pe[ k ] );

		}

		for ( let c = 0; c < featureWidth; c ++ ) {

			activationsStorage.element( actBase.add( int( a0Offset + c ) ) ).assign( a0Vars[ c ] );

		}

		// Append the normalized LOD value as the decoder's final input
		// component (see NTCGridPyramidModel.js's `computeDecoderInputSize`)
		// - necessary because the selected grid level alone doesn't say which
		// of its several covered mips this sample targets; this is what lets
		// the shared decoder disambiguate that, matching the paper's own
		// decoder input layout (Section 4.4: "... and a LOD value").
		activationsStorage.element( actBase.add( int( a0Offset + featureWidth ) ) ).assign( lod.div( Math.max( 1, maxLod ) ) );

		// 2. Forward MLP (hidden layers activated per layer.activation - 'relu'
		// by default, or 'hgelu' - see NTCGridPyramidModel.js's
		// `hiddenActivation` option; linear output).
		for ( let l = 0; l < mlpLayers.length; l ++ ) {

			const layer = mlpLayers[ l ];
			const inBase = actBase.add( int( l === 0 ? a0Offset : layerActs[ l - 1 ].aOffset ) );
			const zBase = actBase.add( int( layerActs[ l ].zOffset ) );
			const aBase = layerActs[ l ].aOffset >= 0 ? actBase.add( int( layerActs[ l ].aOffset ) ) : null;

			forwardDenseLayerTSL( {
				activationsStorage, weightsStorage,
				inputBase: inBase, inputSize: layer.inputSize, outputSize: layer.outputSize,
				weightsOffset: layer.weightsOffset, biasesOffset: layer.biasesOffset,
				zBase, aBase, activation: layer.activation
			} );

		}

		// 3. L2 loss + output delta.
		//
		// Each output channel c may carry its own output nonlinearity (see
		// ./NTCOutputActivations.js, keyed by NeuralMaterialFormat.
		// js's per-channel `activation`) applied on top of this always-linear
		// decoder's raw `z` - the loss is computed
		// against the *activated* prediction `a = activation(z)` (matching a
		// raw-physical-units target), and the stored delta is the chain-rule
		// product `(a - target) * da/dz`, so everything downstream (step 4,
		// the hand-written backward pass) still just consumes a plain
		// per-output `dL/dz` exactly as it did for the old all-linear output.
		//
		// Deltas/gradients are deliberately kept at raw, un-batch-averaged
		// magnitude here (no division by batchSize). Gradients get quantized
		// to fixed-point integers and atomically summed one sample at a time
		// (`int(value * FIXED_POINT_SCALE)` truncates toward zero *before*
		// accumulating - see createTextureAdam*ComputeNode below) - if each
		// individual sample's contribution were pre-divided by batchSize here,
		// most per-sample gradients would truncate to exactly zero well before
		// the network actually converges (an 8192-sample batch needs a raw
		// error of ~8% just to survive 1e-5 quantization once divided by
		// 8192), which reads as loss plateauing into pure noise instead of
		// decreasing further. Dividing by batchSize only happens once, after
		// the full-precision sum has been accumulated (see invBatchUniform
		// usage in the Adam/gradient-norm kernels below).
		const outZBase = actBase.add( int( layerActs[ layerActs.length - 1 ].zOffset ) );
		const outDeltaBase = actBase.add( int( deltaOffsets[ deltaOffsets.length - 1 ] ) );
		const sampleLoss = float( 0.0 ).toVar();

		for ( let c = 0; c < outputChannels; c ++ ) {

			const activation = channelActivations ? channelActivations[ c ] : undefined;
			const z = activationsStorage.element( outZBase.add( c ) );
			const pred = applyChannelActivation( z, activation as any );
			const diff = pred.sub( targetComponents[ c ] );
			sampleLoss.addAssign( diff.mul( diff ).mul( 0.5 ) );
			const delta = diff.mul( channelActivationDerivativeFromOutput( pred, activation as any ) );
			activationsStorage.element( outDeltaBase.add( c ) ).assign( delta );

		}

		addGradient(lossAtomic.element(0), sampleLoss);

		// 4. Backward through the MLP layers (output -> input).
		for ( let l = mlpLayers.length - 1; l >= 0; l -- ) {

			const layer = mlpLayers[ l ];
			const deltaBase = actBase.add( int( deltaOffsets[ l ] ) );
			const inBase = actBase.add( int( l === 0 ? a0Offset : layerActs[ l - 1 ].aOffset ) );

			accumulateDenseLayerGradTSL( {
				activationsStorage, gradWeightsAtomic,
				floatGradients: gpuModel.floatGradients,
				deltaBase, inputBase: inBase, inputSize: layer.inputSize, outputSize: layer.outputSize,
				weightsOffset: layer.weightsOffset, biasesOffset: layer.biasesOffset
			} );

			if ( l > 0 ) {

				const prevZBase = actBase.add( int( layerActs[ l - 1 ].zOffset ) );
				const prevDeltaBase = actBase.add( int( deltaOffsets[ l - 1 ] ) );

				backwardDenseLayerTSL( {
					activationsStorage, weightsStorage,
					deltaBase, deltaSize: layer.outputSize,
					weightsOffset: layer.weightsOffset, prevSize: layer.inputSize,
					prevZBase, outDeltaBase: prevDeltaBase, activation: mlpLayers[ l - 1 ].activation
				} );

			} else {

				// Backward into gradA0 (the concatenated, un-activated grid features).
				const gradA0Base = actBase.add( int( gradA0Offset ) );

				Loop( { start: 0, end: layer.inputSize, type: 'int', name: 'i', condition: '<' }, ( { i }: { i: TSLNode } ) => {

					const gradInput_i = float( 0.0 ).toVar();

					Loop( { start: 0, end: layer.outputSize, type: 'int', name: 'j', condition: '<' }, ( { j }: { j: TSLNode } ) => {

						const delta_j = activationsStorage.element( deltaBase.add( j ) );
						const w_ji = weightsStorage.element( int( layer.weightsOffset ).add( j.mul( layer.inputSize ) ).add( i ) );
						gradInput_i.addAssign( delta_j.mul( w_ji ) );

					} );

					activationsStorage.element( gradA0Base.add( i ) ).assign( gradInput_i );

				} );

			}

		}

		if ( samples.trainLatents !== false ) {
		// 5. Scatter gradA0 back into the latent grids using the same taps
		// computed in the forward pass, gated by the same one-hot selection
		// `weight` (see step 1's comment - only `selectedLevel`'s latents
		// receive nonzero gradient this sample).
		//
		// `positionalEncoding === false` (bilinear): forward computed `a0_c =
		// weight_g * bilinear(...)`, so `d(a0_c)/d(latent) = weight_g *
		// d(bilinear)/d(latent)` - `gradZ_c * weight_g` scattered through the
		// same bilinear taps/weights.
		//
		// `positionalEncoding === true` (raw 4-tap concat): forward computed
		// each of the 4*channels a0 slots as `weight_g * rawTap` directly (no
		// blending), so each slot's gradient scatters 1:1 to its own single
		// latent texel, weighted only by `weight_g` - no `w0..w3` split. The
		// 12 positional-encoding slots need no scatter at all: they're a pure
		// function of `uv` (via `selTx`/`selTy`), not of any trainable
		// latent, so gradA0 there simply isn't read here.
		//
		// `dualGrid`: G1's `channels` slots scatter bilinearly into the
		// coarsest level with no `weight` gate (forward read it
		// unconditionally). When that level is also the selected G0, both
		// slots' gradients land on the same texels - correct, the chain rule
		// sums them.
		const gradA0Base = actBase.add( int( gradA0Offset ) );

		const scatterBilinear = ( taps: typeof levelTaps[ number ], c: number, gradZ_c: TSLNode ) => {

			addGradient( gradLatentsAtomic.element( taps.off0.add( c ) ), gradZ_c.mul( taps.w0 ) );
			addGradient( gradLatentsAtomic.element( taps.off1.add( c ) ), gradZ_c.mul( taps.w1 ) );
			addGradient( gradLatentsAtomic.element( taps.off2.add( c ) ), gradZ_c.mul( taps.w2 ) );
			addGradient( gradLatentsAtomic.element( taps.off3.add( c ) ), gradZ_c.mul( taps.w3 ) );

		};

		for ( let g = 0; g < gridLevels.length; g ++ ) {

			const taps = levelTaps[ g ];

			if (gridLevels[g].isLowRes) {
				for(let c=0;c<lowResChannels;c++) scatterBilinear(taps,c,activationsStorage.element(gradA0Base.add(g0Width+c)).mul(taps.weight));
				continue;
			}

			if ( positionalEncoding ) {

				const offs = [ taps.off0, taps.off1, taps.off2, taps.off3 ];

				for ( let t = 0; t < 4; t ++ ) {

					for ( let c = 0; c < channels; c ++ ) {

						const gradTap_c = activationsStorage.element( gradA0Base.add( t * channels + c ) ).mul( taps.weight );

						addGradient( gradLatentsAtomic.element( offs[ t ].add( c ) ), gradTap_c );

					}

				}

			} else {

				for ( let c = 0; c < channels; c ++ ) scatterBilinear( taps, c, activationsStorage.element( gradA0Base.add( c ) ).mul( taps.weight ) );

			}

		}

		}

	} )().compute( batchSize ).setName( 'NTCTrainBatch' );

}

/**
 * Accumulates squared weight and latent gradients for global norm clipping.
 * Unlike the neural-appearance version, the raw fixed-point sum is first
 * converted back to a batch-averaged gradient (`.mul(invBatchUniform)`)
 * before squaring - see createTextureTrainBatchComputeNode for why gradients
 * are deposited at raw, un-averaged magnitude.
 */
function createAccumulateGradientNormComputeNode( gpuModel: NTCGPUModel, includeLatents = true ): TSLNode {

	const { layout, gradNormAtomic, invBatchUniform } = gpuModel;
	const { gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;
	const { totalWeights, totalLatents } = layout;
	const dispatchCount = totalWeights + ( includeLatents ? totalLatents : 0 );

	return Fn( () => {

		const idx = int( instanceIndex );
		const rawGrad = float( 0.0 ).toVar();

		If( idx.lessThan( int( totalWeights ) ), () => {

			rawGrad.assign(gpuModel.floatGradients ? atomicLoadFloat(gradWeightsAtomic.element(idx)) : float( atomicLoad( gradWeightsAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) ));

		} ).Else( () => {

			const latentIdx = idx.sub( int( totalWeights ) );
			rawGrad.assign(gpuModel.floatGradients ? atomicLoadFloat(gradLatentsAtomic.element(latentIdx)) : float( atomicLoad( gradLatentsAtomic.element( latentIdx ) ) ).div( float( FIXED_POINT_SCALE ) ));

		} );

		const grad = rawGrad.mul( invBatchUniform );
		if (gpuModel.floatGradients) atomicAddFloat(gradNormAtomic.element(0), grad.mul(grad));
		else atomicAdd( gradNormAtomic.element( 0 ), int( grad.mul( grad ).mul( float( GRADIENT_NORM_SCALE ) ) ) );

	} )().compute( dispatchCount ).setName( 'NTCAccumulateGradientNorm' );

}

/**
 * Adam step for MLP weights. Batch-averages the raw fixed-point gradient sum
 * (`.mul(invBatchUniform)`) only after it has been fully accumulated, so
 * individual sample contributions never get quantized below the fixed-point
 * resolution before summing.
 */
function createTextureAdamWeightsComputeNode( gpuModel: NTCGPUModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ): TSLNode {

	const {
		layout,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
	} = gpuModel;
	const { valuesStorage, gradAtomic, mStorage, vStorage } = gpuModel.weightsBuffers;

	return createAdamComputeNode( {
		floatGradients: gpuModel.floatGradients,
		valuesStorage,
		gradAtomic,
		mStorage,
		vStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform: learningRateUniform.mul(gpuModel.weightsLearningRateScale),
		stepUniform,
		invBatchUniform,
		count: layout.totalWeights,
		beta1,
		beta2,
		epsilon,
		name: 'NTCAdamWeights'
	} );

}

/**
 * Adam step for latent grid features - same batch-averaging fix as
 * createTextureAdamWeightsComputeNode above.
 */
function createTextureAdamLatentsComputeNode( gpuModel: NTCGPUModel, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7 } = {} ): TSLNode {

	const {
		layout,
		learningRateUniform,
		stepUniform,
		gradNormAtomic,
		maxGradientNormUniform,
		invBatchUniform
	} = gpuModel;
	const { valuesStorage, gradAtomic, mStorage, vStorage } = gpuModel.latentsBuffers;

	return createAdamComputeNode( {
		floatGradients: gpuModel.floatGradients,
		valuesStorage,
		gradAtomic,
		mStorage,
		vStorage,
		gradNormAtomic,
		maxGradientNormUniform,
		learningRateUniform,
		stepUniform,
		invBatchUniform,
		count: layout.totalLatents,
		beta1,
		beta2,
		epsilon,
		name: 'NTCAdamLatents',
		transform: gpuModel.quantization.mode !== 'none' && gpuModel.quantization.range !== 'auto'
			? (value: any) => value.clamp(gpuModel.quantization.range[0],gpuModel.quantization.range[1]) : undefined
	} );

}

export {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
};
