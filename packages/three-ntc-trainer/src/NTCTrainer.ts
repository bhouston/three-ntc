import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { DEFAULT_MIPS_PER_LEVEL } from './NTCGridModel.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import {
	createTextureTrainBatchComputeNode,
	createAccumulateGradientNormComputeNode,
	createTextureAdamWeightsComputeNode,
	createTextureAdamLatentsComputeNode
} from './NTCGPUComputeTSL.js';
import { createResetGradientNormComputeNode } from './NTCGPUKernelsTSL.js';
import { getLearningRate, createRandom, yieldToBrowser } from './NTCTrainingUtils.js';
import { DEFAULT_QUANTIZATION_OPTIONS, QUANTIZATION_SCHEMES, resolveQuantizationConfig, refreshGPUQuantizationRange } from './NTCQuantization.js';

// How often (in training iterations) `quantization.range === 'auto'` is
// re-measured from the live GPU latent buffer (see
// NeuralQuantization.refreshGPUQuantizationRange). Latents move gradually
// under Adam, so the true min/max drifts slowly - a readback every 64
// iterations is frequent enough to track that drift without ever clipping
// noticeably stale, while staying rare enough (a full GPU->CPU latent-buffer
// round trip each time) not to meaningfully slow down training, which
// already only syncs that often anyway (see the `shouldSync`/`onProgress`
// cadence below).
const QUANTIZATION_RANGE_REFRESH_INTERVAL = 64;

/**
 * Reads the largest dimension off the first source texture that actually has
 * one available - a plain `Texture.image` (`{ width, height }`, the loaded-
 * image path) or a render-target texture's `.image` (set by
 * `THREE.RenderTarget` to its own `width`/`height`, the baked-color-node
 * path - see NTCTextureSource.js). Returns `null` (rather than throwing) when
 * no source texture exposes a usable size, in which case the caller falls
 * back to `createNTCGridPyramidModel`'s own fallback instead.
 */
function resolveSourceTextureResolution( textures: any[] ): number | null {

	for ( const texture of textures ) {

		const image = texture.image;
		if ( image && image.width && image.height ) return Math.max( image.width, image.height );

	}

	return null;

}

const DEFAULT_OPTIONS = {
	// Feature-vector width per grid cell (the paper's "grid channels") - see
	// NTCGridPyramidModel.js.
	gridChannels: 4,
	levels: 4,
	baseResolution: 128,
	// How many physical mip levels each stored grid level is reused to
	// reconstruct - see NTCGridModel.js/NTCMipBands.js. Training is always
	// mip-aware (there is no "off" switch): every model is trained across its
	// whole physical mip range (see `textureResolution`/`maxLod` below), not
	// just mip 0, so it anti-aliases correctly when viewed from a distance.
	mipsPerLevel: DEFAULT_MIPS_PER_LEVEL,
	hiddenSizes: [ 32, 32 ],
	// See NTCGridPyramidModel.js's `hiddenActivation` option - 'relu' (the
	// default) or 'hgelu' (the NVIDIA neural texture compression paper's own
	// cheap GELU approximation, Section 4.4).
	hiddenActivation: 'relu',
	// Optional (default off) - see NTCGridPyramidModel.js's
	// `computeDecoderInputSize` doc comment: the NVIDIA neural texture
	// compression paper's Section 4.3 decoder input (4 concatenated raw
	// neighbor taps + 12 tiled positional-encoding scalars) instead of this
	// addon's original plain bilinear tap. Larger/slower decoder input, may
	// reconstruct sharper sub-grid-resolution detail.
	positionalEncoding: false,
	// Optional (default off) - the paper's G0/G1 grid pair, see
	// NTCGridPyramidModel.js's `computeDecoderInputSize` doc comment.
	dualGrid: false,
	outputChannels: 3,
	batchSize: 4096,
	learningRate: 0.01,
	// Anneals all the way to (near) zero by the final iteration. A learning
	// rate that never fully decays keeps Adam injecting per-step gradient
	// noise into individual grid texels indefinitely, which - since each
	// texel only sees a handful of samples per iteration - doesn't average
	// out and shows up as a noise floor that stops shrinking with more
	// training. NVIDIA's neural texture compression trainer anneals the
	// same way (cosine schedule to 0).
	cosineAnnealingScale: 0.001,
	iterations: 3000,
	maxGradientNorm: 1,
	seed: 1,
	name: 'trained neural texture',
	// Quantization-Aware Training (QAT) of the latent grid - see
	// NeuralQuantization.js. Defaults to `mode: 'none'` (a byte-for-byte
	// no-op vs. training without QAT at all).
	quantization: DEFAULT_QUANTIZATION_OPTIONS,
	// Fraction of `iterations` run *after* training with the latent grid
	// hard-quantized and frozen, so only the MLP keeps updating and learns to
	// absorb the real rounding error (the paper's post-quantization retrain,
	// 5%). Ignored when `quantization.mode` is 'none'.
	retrainAfterQuantize: 0.05
};

interface NTCTrainerOptions {
	gridChannels?: number;
	levels?: number;
	baseResolution?: number;
	mipsPerLevel?: number;
	hiddenSizes?: number[];
	hiddenActivation?: string;
	positionalEncoding?: boolean;
	dualGrid?: boolean;
	outputChannels?: number;
	batchSize?: number;
	learningRate?: number;
	/** Optional separate MLP rate, decayed proportionally with the latent rate. */
	weightsLearningRate?: number;
	cosineAnnealingScale?: number;
	iterations?: number;
	maxGradientNorm?: number;
	seed?: number;
	name?: string;
	quantization?: typeof DEFAULT_QUANTIZATION_OPTIONS;
	retrainAfterQuantize?: number;
	textureResolution?: number;
	channelActivations?: string[];
	uvTransform?: any;
	[key: string]: unknown;
}

interface NTCTrainProgress {
	iteration: number;
	iterations: number;
	loss: number;
	learningRate: number;
	cpuModel: any;
	gpuModel: NTCGPUModel;
}

interface NTCTrainArgs {
	renderer: any;
	sourceTexture?: any;
	sourceTextures?: any[];
	onProgress?: ( ( progress: NTCTrainProgress ) => void ) | null;
}

/**
 * Browser-side trainer that fits a small mip-pyramid-of-feature-grids + MLP
 * neural representation to a single static GPU texture (e.g. a material's
 * albedo/base-color channel), following the NVIDIA neural texture
 * compression recipe: a handful of trainable feature grids, each reused (via
 * an explicit LOD input) to reconstruct several physical mip levels, feeding
 * a shallow shared MLP decoder, trained with Adam + L2 loss.
 *
 * Unlike `NeuralAppearanceTrainer` (which distills an entire BRDF's shading
 * response and therefore needs a rendered "teacher atlas" + CPU readback),
 * the teacher here is already a static GPU texture, so training samples are
 * generated and read entirely on the GPU inside the training compute
 * shader - no readback round trip is needed per iteration.
 */
class NTCTrainer {

	options: NTCTrainerOptions;
	random: () => number;
	quantizationRange: Array<[ number, number ]> | null;
	private _abortRequested: boolean;

	constructor( options: NTCTrainerOptions = {} ) {

		this.options = { ...DEFAULT_OPTIONS, ...options };
		this.random = createRandom( this.options.seed as number );
		this._abortRequested = false;
		this.quantizationRange = null;

	}

	/**
	 * Requests that an in-flight `train()` call finish after the current
	 * iteration and return the current model, as if the run had completed.
	 */
	abort(): void {

		this._abortRequested = true;

	}

	/**
	 * Hard-rounds the live GPU latent grid to its quantization levels (the
	 * same `quantizeForwardCPU` the export codec mirrors) against the final
	 * `'auto'` range, and re-uploads it. The caller then stops running the
	 * latent Adam step, so the grid is exactly what will be written to disk
	 * while the MLP keeps training against it.
	 */
	private async _quantizeAndFreezeLatents( gpuModel: NTCGPUModel, renderer: any, quantization: ReturnType<typeof resolveQuantizationConfig> ): Promise<void> {

		if ( quantization.range === 'auto' ) await refreshGPUQuantizationRange( gpuModel, renderer );

		const ranges = gpuModel.getQuantizationRange();
		const quantize = QUANTIZATION_SCHEMES[ quantization.mode ].quantizeForwardCPU;
		const latents = new Float32Array( await renderer.getArrayBufferAsync( gpuModel.latentsBuffers.attribute ) );

		gpuModel.layout.gridLevels.forEach( ( level: any, g: number ) => {

			const [ lo, hi ] = ranges[ g ];
			for ( let i = level.offset; i < level.offset + level.floatCount; i ++ ) latents[ i ] = quantize( latents[ i ], lo, hi );

		} );

		( gpuModel.latentsBuffers.attribute.array as Float32Array ).set( latents );
		gpuModel.latentsBuffers.attribute.needsUpdate = true;

	}

	async train( { renderer, sourceTexture, sourceTextures, onProgress = null }: NTCTrainArgs ) {

		const settings = this.options;
		this._abortRequested = false;

		if ( ! renderer || renderer.isWebGPURenderer !== true ) {

			throw new Error( 'THREE.NTCTrainer: WebGPU renderer is required for neural texture training.' );

		}

		const textures = sourceTextures || ( sourceTexture ? [ sourceTexture ] : null );

		if ( ! textures || textures.length === 0 ) {

			throw new Error( 'THREE.NTCTrainer: a sourceTexture (or sourceTextures array) is required.' );

		}

		// Validated once up front (throws a clear error immediately rather
		// than deep inside GPUModel construction) - `gpuModel` below resolves
		// it again from the same `settings.quantization` input, which is
		// idempotent (see resolveQuantizationConfig's doc comment).
		const quantization = resolveQuantizationConfig( settings as any );
		this.quantizationRange = null;

		// Training needs the real source texture resolution to know the full
		// physical mip range (`maxLod`) this model should support - resolved
		// here from the actual GPU texture(s) rather than left to
		// createNTCGridPyramidModel's fallback (the finest grid resolution),
		// which is only ever a stand-in for when the true texture size isn't
		// known. An explicit `settings.textureResolution` always wins.
		const modelSettings = settings.textureResolution ? settings : {
			...settings,
			textureResolution: resolveSourceTextureResolution( textures ) || undefined
		};

		const cpuModel = createNTCGridPyramidModel( modelSettings, this.random );
		const gpuModel = new NTCGPUModel( modelSettings );
		gpuModel.initFromCPUModel( cpuModel );

		try {

			const trainBatchNode = createTextureTrainBatchComputeNode( gpuModel, textures );
			const frozenBatchNode = createTextureTrainBatchComputeNode( gpuModel, textures, { trainLatents: false } );
			const frozenNormNode = createAccumulateGradientNormComputeNode( gpuModel, false );
			const resetGradientNormNode = createResetGradientNormComputeNode( gpuModel );
			const accumulateGradientNormNode = createAccumulateGradientNormComputeNode( gpuModel );
			const adamWeightsNode = createTextureAdamWeightsComputeNode( gpuModel );
			const adamLatentsNode = createTextureAdamLatentsComputeNode( gpuModel );

			const trainIterations = settings.iterations as number;
			const retrainIterations = quantization.mode === 'none' ? 0 :
				Math.round( trainIterations * ( settings.retrainAfterQuantize as number ) );
			const iterations = trainIterations + retrainIterations;
			// Retrain phase: its own short cosine schedule at a tenth of the
			// base rate - a full-rate restart on the MLP alone would undo more
			// than the frozen rounding error it's meant to absorb.
			const retrainSettings = { ...settings, iterations: retrainIterations, learningRate: ( settings.learningRate as number ) * 0.1 };
			let lastLoss = NaN;
			let completedIterations = 0;
			let latentsFrozen = false;

			for ( let iteration = 0; iteration < iterations; iteration ++ ) {

				if ( this._abortRequested ) break;

				if ( iteration === trainIterations ) {

					await this._quantizeAndFreezeLatents( gpuModel, renderer, quantization );
					latentsFrozen = true;

				}

				const learningRate = latentsFrozen ?
					getLearningRate( retrainSettings as any, iteration - trainIterations ) :
					getLearningRate( settings as any, iteration );
				gpuModel.resetLoss();
				gpuModel.learningRateUniform.value = learningRate;
				gpuModel.stepUniform.value = iteration + 1;
				gpuModel.maxGradientNormUniform.value = settings.maxGradientNorm;

				renderer.compute( latentsFrozen ? frozenBatchNode : trainBatchNode );
				renderer.compute( resetGradientNormNode );
				renderer.compute( latentsFrozen ? frozenNormNode : accumulateGradientNormNode );
				renderer.compute( adamWeightsNode );
				if ( ! latentsFrozen ) renderer.compute( adamLatentsNode );

				completedIterations = iteration + 1;

				const shouldSync = onProgress !== null && ( iteration % 4 === 0 || iteration === iterations - 1 );

				if ( shouldSync ) {

					[ lastLoss ] = await Promise.all( [
						gpuModel.readLoss( renderer ),
						gpuModel.syncToCPU( cpuModel, renderer )
					] );

					if ( onProgress ) {

						onProgress( { iteration: completedIterations, iterations, loss: lastLoss, learningRate, cpuModel, gpuModel } );

					}

					await yieldToBrowser();

				}

				// QAT `range: 'auto'` periodic re-measurement (see
				// NeuralQuantization.refreshGPUQuantizationRange and
				// QUANTIZATION_RANGE_REFRESH_INTERVAL's doc comment above).
				// Gated on `quantization` (resolved once above from
				// `settings`, not `gpuModel.quantization`) so the default
				// `mode: 'none'` path never touches `gpuModel` for this at
				// all - a true no-op, not just a cheap early-return.
				if ( quantization.mode !== 'none' && quantization.range === 'auto' && ! latentsFrozen &&
					( iteration % QUANTIZATION_RANGE_REFRESH_INTERVAL === QUANTIZATION_RANGE_REFRESH_INTERVAL - 1 || iteration === trainIterations - 1 ) ) {

					await refreshGPUQuantizationRange( gpuModel, renderer );

				}

				if ( iteration % 32 === 31 ) await yieldToBrowser();

			}

			await gpuModel.syncToCPU( cpuModel, renderer );

			// Freeze the final QAT range (see NeuralQuantization.js) so a
			// later export phase can quantize the exported latent grid against
			// exactly the range the network was actually trained/evaluated
			// against - not a range measured before the last few optimizer
			// steps moved the latents further. Retrievable as either
			// `trainer.quantizationRange` (this trainer instance) or
			// `cpuModel.quantizationRange` (the returned model) - both are the
			// same `[min, max]`-per-level array, or `null` when quantization is
			// disabled (`mode: 'none'`).
			if ( quantization.mode !== 'none' ) {

				if ( quantization.range === 'auto' && ! latentsFrozen ) await refreshGPUQuantizationRange( gpuModel, renderer );
				this.quantizationRange = gpuModel.getQuantizationRange();

			}

			cpuModel.quantizationRange = this.quantizationRange;
			// Read by NTCManifest.encodeNTC to pick the on-disk latent dtype.
			cpuModel.quantization = quantization;

			return {
				cpuModel,
				loss: lastLoss,
				iteration: completedIterations,
				iterations,
				stoppedEarly: completedIterations < iterations,
				quantization,
				quantizationRange: this.quantizationRange
			};

		} finally {

			gpuModel.dispose();

		}

	}

}

export { NTCTrainer };
export type { NTCTrainerOptions, NTCTrainProgress, NTCTrainArgs };
