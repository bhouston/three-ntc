import { computeFeatureLodOffset } from 'three-ntc';
import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform } from 'three/tsl';
import { FIXED_POINT_SCALE } from './NTCGPUTrainingConstants.js';
import { createAdamParameterBuffers, disposeAdamParameterBuffers, type AdamParameterBuffers, type TSLNode } from './NTCGPUKernelsTSL.js';
import { computeGridLevels } from './NTCGridModel.js';
import { resolveNTCGridPyramidOptions, computeDecoderInputSize, type NTCGridPyramidOptions } from './NTCGridPyramidModel.js';
import { resolveQuantizationConfig, type ResolvedNTCQuantizationConfig, type NTCQuantizationOptions } from './NTCQuantization.js';

interface GridLevelLayout {
	channels: number;
	featureLevel: number;
	isLowRes: boolean;
	width: number;
	height: number;
	offset: number;
	texelCount: number;
	floatCount: number;
}

interface MLPLayerLayout {
	inputSize: number;
	outputSize: number;
	weightsOffset: number;
	weightsCount: number;
	biasesOffset: number;
	biasesCount: number;
	isOutput: boolean;
	activation: string;
}

interface LayerActs {
	zOffset: number;
	aOffset: number;
}

interface NTCTextureModelLayout {
	lowResChannels: number;
	channels: number;
	levels: number;
	mipsPerLevel: number;
	resolutions: number[];
	hiddenSizes: number[];
	hiddenActivation: string;
	outputChannels: number;
	channelActivations: string[] | null;
	textureResolution: number;
	maxLod: number;
	lodOffset: number;
	positionalEncodingPeriod: number;
	inputSize: number;
	positionalEncoding: boolean;
	dualGrid: boolean;
	gridLevels: GridLevelLayout[];
	totalLatents: number;
	mlpLayers: MLPLayerLayout[];
	totalWeights: number;
	a0Offset: number;
	layerActs: LayerActs[];
	deltaOffsets: number[];
	gradA0Offset: number;
	activationStride: number;
}

interface NTCGPUModelOptions extends NTCGridPyramidOptions {
	batchSize?: number;
	learningRate?: number;
	weightsLearningRate?: number;
	maxGradientNorm?: number;
	channelActivations?: string[];
	quantization?: NTCQuantizationOptions;
}

/**
 * Computes buffer layouts and offsets for GPU-based neural texture training:
 * a genuine mip pyramid of feature grids plus a small MLP decoder, sized
 * generically from an arbitrary hidden-layer configuration.
 *
 * Mirrors `createNTCGridPyramidModel` (NTCGridPyramidModel.js) exactly for
 * `inputSize` - both derive it from the same `resolveNTCGridPyramidOptions`,
 * so a trainer's CPU model and its GPU buffer layout can never disagree about
 * the decoder's input shape.
 */
function computeTextureModelLayout( options: NTCGPUModelOptions = {} ): NTCTextureModelLayout {

	const { channels, lowResChannels, levels: requestedLevels, baseResolution, mipsPerLevel, hiddenSizes, hiddenActivation, outputChannels, textureResolution, positionalEncoding, dualGrid } = resolveNTCGridPyramidOptions( options );
	// One entry per output channel naming its output nonlinearity (see
	// ./NTCOutputActivations.js); undefined/omitted entries (the
	// default, `options.channelActivations` unset) mean plain linear, i.e.
	// today's behavior for every neural-texture caller that doesn't pass
	// this - only neural-material does (see NeuralMaterialFormat.js).
	const channelActivations = options.channelActivations || null;

	const resolutions = computeGridLevels( baseResolution, requestedLevels, mipsPerLevel );
	const levels = resolutions.length;

	// Mirrors NTCGridPyramidModel.js's `maxLod` derivation exactly - see its
	// doc comment.
	const resolvedTextureResolution = textureResolution || resolutions[ 0 ];
	const lodOffset = computeFeatureLodOffset(resolvedTextureResolution,resolutions[0]);
	const maxLod = Math.floor( Math.log2( Math.max( 1, resolvedTextureResolution ) ) );

	const gridLevels: GridLevelLayout[] = [];
	let latentOffset = 0;

	for ( const isLowRes of (dualGrid ? [false,true] : [false]) ) {
		resolutions.forEach((r,featureLevel) => {
			const resolution = isLowRes ? Math.max(1,Math.floor(r/2)) : r;
			const width = isLowRes ? lowResChannels : channels;
			const texelCount = resolution * resolution;
			const floatCount = texelCount * width;
			gridLevels.push({width:resolution,height:resolution,channels:width,featureLevel,isLowRes,offset:latentOffset,texelCount,floatCount});
			latentOffset += floatCount;
		});
	}

	const totalLatents = latentOffset;

	// MLP weight layout: input = one grid level's tap(s) (the level selected
	// by this sample's LOD, see NTCGPUComputeTSL.js step 1) plus the
	// normalized LOD itself - fixed width regardless of how many mip levels
	// the pyramid has (see NTCGridPyramidModel.js's `computeDecoderInputSize`
	// doc comment for the two widths: plain bilinear tap, or - when
	// `positionalEncoding` is on - 4 concatenated raw taps + positional
	// encoding).
	const inputSize = computeDecoderInputSize( channels, positionalEncoding, dualGrid, lowResChannels );
	const sizes = [ inputSize, ...hiddenSizes, outputChannels ];
	const mlpLayers: MLPLayerLayout[] = [];
	let weightOffset = 0;

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const inSize = sizes[ i ];
		const outSize = sizes[ i + 1 ];
		const weightsOffset = weightOffset;
		const weightsCount = inSize * outSize;
		const biasesOffset = weightsOffset + weightsCount;
		const biasesCount = outSize;
		weightOffset = biasesOffset + biasesCount;

		const isOutput = i === sizes.length - 2;

		mlpLayers.push( {
			inputSize: inSize,
			outputSize: outSize,
			weightsOffset,
			weightsCount,
			biasesOffset,
			biasesCount,
			isOutput,
			// Mirrors NTCMLP.js's createMLP: every hidden layer uses
			// `hiddenActivation` ('relu' by default, or 'hgelu'), the output
			// layer is always linear - matched here so
			// NTCGPUComputeTSL.js's forward/backward passes apply the same
			// activation (and its derivative) the CPU reference model does.
			activation: isOutput ? 'linear' : hiddenActivation
		} );

	}

	const totalWeights = weightOffset;

	// Per-sample activation buffer layout (forward a0/z/a, backward deltas, gradA0).
	let cursor = 0;
	const a0Offset = cursor; cursor += inputSize;
	const layerActs: LayerActs[] = [];

	for ( const layer of mlpLayers ) {

		const zOffset = cursor; cursor += layer.outputSize;
		let aOffset = - 1;

		if ( layer.isOutput === false ) {

			aOffset = cursor; cursor += layer.outputSize;

		}

		layerActs.push( { zOffset, aOffset } );

	}

	const deltaOffsets: number[] = [];

	for ( const layer of mlpLayers ) {

		deltaOffsets.push( cursor );
		cursor += layer.outputSize;

	}

	const gradA0Offset = cursor; cursor += inputSize;
	const activationStride = cursor;

	return {
		channels,
		lowResChannels,
		levels,
		mipsPerLevel,
		resolutions,
		hiddenSizes,
		hiddenActivation,
		outputChannels,
		channelActivations,
		textureResolution: resolvedTextureResolution,
		maxLod,
		lodOffset,
		positionalEncodingPeriod: options.positionalEncodingPeriod ?? 0,
		inputSize,
		positionalEncoding,
		dualGrid,
		gridLevels,
		totalLatents,
		mlpLayers,
		totalWeights,
		a0Offset,
		layerActs,
		deltaOffsets,
		gradA0Offset,
		activationStride
	};

}

/**
 * Copies weights/biases and latent-grid data between the CPU reference
 * model and a flat GPU-layout-shaped array, in either direction - the two
 * are identical index-for-index copy loops with only the assignment side
 * flipped, so `initFromCPUModel`/`syncToCPU` both delegate here rather than
 * maintaining that loop twice.
 */
function copyModel( cpuModel: any, layout: NTCTextureModelLayout, weights: Float32Array, latents: Float32Array, direction: 'toGPU' | 'toCPU' ): void {

	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const layerLayout = layout.mlpLayers[ l ];

		if ( direction === 'toGPU' ) {

			for ( let i = 0; i < layer.weights.length; i ++ ) weights[ layerLayout.weightsOffset + i ] = layer.weights[ i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) weights[ layerLayout.biasesOffset + i ] = layer.biases[ i ];

		} else {

			for ( let i = 0; i < layer.weights.length; i ++ ) layer.weights[ i ] = weights[ layerLayout.weightsOffset + i ];
			for ( let i = 0; i < layer.biases.length; i ++ ) layer.biases[ i ] = weights[ layerLayout.biasesOffset + i ];

		}

	}

	const grids = [...cpuModel.grids, ...(cpuModel.lowResGrids || [])];
	for ( let g = 0; g < grids.length; g ++ ) {

		const grid = grids[ g ];
		const level = layout.gridLevels[ g ];

		if ( direction === 'toGPU' ) {

			for ( let i = 0; i < grid.data.length; i ++ ) latents[ level.offset + i ] = grid.data[ i ];

		} else {

			for ( let i = 0; i < grid.data.length; i ++ ) grid.data[ i ] = latents[ level.offset + i ];

		}

	}

}

/**
 * Encapsulates the GPU StorageBuffers, uniforms, and CPU synchronizers for
 * neural texture training. `weightsBuffers`/`latentsBuffers` shapes
 * intentionally mirror `NeuralAppearanceGPUModel`'s so the Adam / gradient-
 * clip compute kernels can be reused verbatim.
 */
class NTCGPUModel {

	options: NTCGPUModelOptions;
	batchSize: number;
	layout: NTCTextureModelLayout;
	weightsBuffers: AdamParameterBuffers;
	latentsBuffers: AdamParameterBuffers;
	activationsAttribute: InstanceType<typeof StorageBufferAttribute>;
	activationsStorage: TSLNode;
	lossAttribute: InstanceType<typeof StorageBufferAttribute>;
	lossAtomic: TSLNode;
	gradNormAttribute: InstanceType<typeof StorageBufferAttribute>;
	gradNormAtomic: TSLNode;
	invBatchUniform: TSLNode;
	learningRateUniform: TSLNode;
	weightsLearningRateScale: number;
	quantizationNoiseUniform: TSLNode;
	stepUniform: TSLNode;
	maxGradientNormUniform: TSLNode;
	quantization: ResolvedNTCQuantizationConfig;
	quantizationRangeUniforms: Array<{ min: TSLNode; max: TSLNode }>;

	constructor( options: NTCGPUModelOptions = {} ) {

		this.options = options;
		this.batchSize = options.batchSize || 4096;
		this.layout = computeTextureModelLayout( options );

		const { totalWeights, totalLatents, activationStride } = this.layout;
		const batchSize = this.batchSize;

		this.weightsBuffers = createAdamParameterBuffers( totalWeights );
		this.latentsBuffers = createAdamParameterBuffers( totalLatents );

		this.activationsAttribute = new StorageBufferAttribute( new Float32Array( batchSize * activationStride ), 1, Float32Array );
		this.activationsStorage = storage( this.activationsAttribute, 'float', batchSize * activationStride );

		this.lossAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.lossAtomic = storage( this.lossAttribute, 'int', 1 ).toAtomic();

		this.gradNormAttribute = new StorageBufferAttribute( new Int32Array( 1 ), 1, Int32Array );
		this.gradNormAtomic = storage( this.gradNormAttribute, 'int', 1 ).toAtomic();

		this.invBatchUniform = uniform( 1.0 / batchSize );
		this.learningRateUniform = uniform( options.learningRate ?? 0.01 );
		this.weightsLearningRateScale = options.weightsLearningRate === undefined ? 1 : options.weightsLearningRate / (options.learningRate ?? 0.01);
		if (!Number.isFinite(this.weightsLearningRateScale) || this.weightsLearningRateScale < 0) throw new Error("Invalid weightsLearningRate / learningRate ratio");
		this.stepUniform = uniform( 1 );
		this.maxGradientNormUniform = uniform( options.maxGradientNorm || 1 );

		// One quantization range per stored grid, shared by tap sampling and export.
		// Noise uses a fixed zero-aligned range; legacy STE can track auto ranges.
		this.quantization = resolveQuantizationConfig( options );
		this.quantizationNoiseUniform = uniform(this.quantization.method === 'noise' ? 1 : 0);
		this.quantizationRangeUniforms = this.layout.gridLevels.map( () => ( {
			min: uniform( this.quantization.range === 'auto' ? - 1 : this.quantization.range[ 0 ] ),
			max: uniform( this.quantization.range === 'auto' ? 1 : this.quantization.range[ 1 ] )
		} ) );

	}

	/**
	 * Updates every per-level quantization-range uniform from a freshly
	 * computed `[min, max]` tuple array (see
	 * `NeuralQuantization.computeLatentRanges`) - called periodically (and
	 * once more at the very end) by NTCTrainer.js when
	 * `quantization.range === 'auto'`.
	 */
	setQuantizationRange( ranges: Array<[ number, number ]> ): void {

		for ( let g = 0; g < this.quantizationRangeUniforms.length; g ++ ) {

			this.quantizationRangeUniforms[ g ].min.value = ranges[ g ][ 0 ];
			this.quantizationRangeUniforms[ g ].max.value = ranges[ g ][ 1 ];

		}

	}

	/**
	 * Reads back the current per-level quantization range as plain
	 * `[min, max]` arrays (not TSL uniform nodes) - used to freeze the final
	 * range at the end of training (see NTCTrainer.js).
	 */
	getQuantizationRange(): Array<[ number, number ]> {

		return this.quantizationRangeUniforms.map( ( { min, max } ) => [ min.value, max.value ] as [ number, number ] );

	}

	initFromCPUModel( cpuModel: any ): void {

		const weights = this.weightsBuffers.attribute.array as Float32Array;
		const latents = this.latentsBuffers.attribute.array as Float32Array;
		weights.fill( 0 );
		latents.fill( 0 );

		copyModel( cpuModel, this.layout, weights, latents, 'toGPU' );
		if (this.quantization.mode !== 'none' && this.quantization.range !== 'auto') {
			const [lo,hi]=this.quantization.range;
			for(let i=0;i<latents.length;i++) latents[i]=Math.min(hi,Math.max(lo,latents[i]));
		}


		this.weightsBuffers.attribute.needsUpdate = true;
		this.latentsBuffers.attribute.needsUpdate = true;

	}

	resetLoss(): void {

		( this.lossAttribute.array as Int32Array )[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

	}

	async syncToCPU( cpuModel: any, renderer: any ): Promise<void> {

		const [ weightsBuffer, latentsBuffer ] = await Promise.all( [
			renderer.getArrayBufferAsync( this.weightsBuffers.attribute ),
			renderer.getArrayBufferAsync( this.latentsBuffers.attribute )
		] );
		const weights = new Float32Array( weightsBuffer );
		const latents = new Float32Array( latentsBuffer );

		copyModel( cpuModel, this.layout, weights, latents, 'toCPU' );

	}

	/**
	 * Releases every GPU storage buffer this instance owns. Training GPU
	 * models are cheap to construct but hold 11 real GPU-side buffers - a
	 * fresh trainer run that doesn't dispose the previous one leaks them.
	 */
	dispose(): void {

		disposeAdamParameterBuffers( this.weightsBuffers );
		disposeAdamParameterBuffers( this.latentsBuffers );

		this.activationsAttribute.dispose();
		this.lossAttribute.dispose();
		this.gradNormAttribute.dispose();

	}

	async readLoss( renderer: any ): Promise<number> {

		const buffer = await renderer.getArrayBufferAsync( this.lossAttribute );
		const array = new Int32Array( buffer );
		// The kernel accumulates the raw (un-batch-averaged) per-sample loss
		// sum - see NeuralTextureGPUComputeTSL.js for why - so the mean loss is
		// recovered here by dividing by batchSize as well as FIXED_POINT_SCALE.
		const loss = array[ 0 ] / ( FIXED_POINT_SCALE * this.batchSize );

		( this.lossAttribute.array as Int32Array )[ 0 ] = 0;
		this.lossAttribute.needsUpdate = true;

		return loss;

	}

}

export { computeTextureModelLayout, NTCGPUModel };
export type { NTCTextureModelLayout, NTCGPUModelOptions, GridLevelLayout as NTCTextureGridLevelLayout };
