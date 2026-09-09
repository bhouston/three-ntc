import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	createMat4Storage,
	createVec4Storage
} from './NTCMLPTSL.js';
import { buildMipChainTexture, NTCGrid } from './NTCHalfFloatTexture.js';
import { MLPLayer } from './NTCBinaryCodec.js';
import { float, textureLevel } from 'three/tsl';

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
 */
function evaluateNeuralTextureRaw( uvNode: any, cpuModel: NTCCpuModel, mipChainTexture: any, renderer: any | null = null, lodNode: any = null ): any[] {

	// renderer is accepted for compatibility with the higher-level material
	// constructor, but this stock-Three.js path always uses fp32 uniforms.
	void renderer;

	const resolvedLodNode = lodNode || float( 0 );
	const channels = cpuModel.channels;

	const sample = textureLevel( mipChainTexture, uvNode, resolvedLodNode );
	const features: any[] = [ sample.x, sample.y, sample.z, sample.w ].slice( 0, channels );

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `inputSize = channels + 1` /
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

export { evaluateNeuralTextureRaw, buildMipChainTexture };
