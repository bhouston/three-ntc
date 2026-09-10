/**
 * Logical model payload estimates. Runtime uses FP16 grid values and FP32
 * decoder parameters; excludes GPU alignment/padding, CPU copies, and training
 * buffers. Storage counts packed latent values plus FP16 decoder parameters,
 * before base64 expansion and JSON metadata (not the exported file size).
 */

const GRID_MEMORY_BYTES_PER_CHANNEL = 2; // fp16 runtime latent texture (NeuralHalfFloatTexture.js)
const GRID_STORAGE_BYTES_PER_CHANNEL = 1; // uint8-quantized export (NeuralBinaryCodec.encodeUint8Base64)
const MLP_MEMORY_BYTES_PER_PARAM = 4; // fp32 runtime weight buffers/uniforms
const MLP_STORAGE_BYTES_PER_PARAM = 2; // float16-packed export (NeuralBinaryCodec.encodeFloat16Base64)

function formatBytes( bytes: number ): string {

	if ( bytes < 1024 ) return `${bytes} B`;
	if ( bytes < 1024 * 1024 ) return `${( bytes / 1024 ).toFixed( 1 )} KiB`;
	return `${( bytes / ( 1024 * 1024 ) ).toFixed( 1 )} MiB`;

}

// Formats a FLOP count with the same K/M/G scaling as formatBytes, e.g.
// `formatFlops( 12345 )` -> "12.3K FLOP".
function formatFlops( flops: number ): string {

	if ( flops < 1000 ) return `${flops} FLOP`;
	if ( flops < 1000 * 1000 ) return `${( flops / 1000 ).toFixed( 1 )}K FLOP`;
	if ( flops < 1000 * 1000 * 1000 ) return `${( flops / ( 1000 * 1000 ) ).toFixed( 2 )}M FLOP`;
	return `${( flops / ( 1000 * 1000 * 1000 ) ).toFixed( 2 )}G FLOP`;

}

// Parameter count / FLOPs for a single dense (fully-connected) layer mapping
// `inputSize` -> `outputSize`, optionally without a bias term - most layers
// have one, but e.g. neural-appearance's BRDF-frame rotation matrix is a
// bias-free linear layer (see NeuralAppearanceManifest.js's `rotation`
// block). FLOPs counts one multiply + one add per weight (the standard
// "2x parameter count" convention for a forward matmul) plus one add per
// bias.
function computeLinearParamCount( inputSize: number, outputSize: number, hasBias = true ): number {

	return inputSize * outputSize + ( hasBias ? outputSize : 0 );

}

function computeLinearFlops( inputSize: number, outputSize: number, hasBias = true ): number {

	return 2 * inputSize * outputSize + ( hasBias ? outputSize : 0 );

}

interface MLPShapeSpec {
	inputSize: number;
	hiddenSize: number;
	hiddenLayers?: number;
	outputSize: number;
}

// Weight + bias parameter count for a fully-connected MLP with `hiddenLayers`
// hidden layers of width `hiddenSize`, mapping `inputSize` to `outputSize`.
function computeMLPParamCount( { inputSize, hiddenSize, hiddenLayers = 2, outputSize }: MLPShapeSpec ): number {

	if ( hiddenLayers <= 0 ) return computeLinearParamCount( inputSize, outputSize );

	let count = computeLinearParamCount( inputSize, hiddenSize );
	for ( let i = 1; i < hiddenLayers; i ++ ) count += computeLinearParamCount( hiddenSize, hiddenSize );
	count += computeLinearParamCount( hiddenSize, outputSize );

	return count;

}

// FLOPs for one forward pass through the same MLP shape as
// `computeMLPParamCount` (same `{ inputSize, hiddenSize, hiddenLayers,
// outputSize }` spec) - i.e. the cost of evaluating this decoder once, for
// one texel/pixel.
function computeMLPFlops( { inputSize, hiddenSize, hiddenLayers = 2, outputSize }: MLPShapeSpec ): number {

	if ( hiddenLayers <= 0 ) return computeLinearFlops( inputSize, outputSize );

	let flops = computeLinearFlops( inputSize, hiddenSize );
	for ( let i = 1; i < hiddenLayers; i ++ ) flops += computeLinearFlops( hiddenSize, hiddenSize );
	flops += computeLinearFlops( hiddenSize, outputSize );

	return flops;

}

// Total latent texel count across a multiresolution grid's per-level square
// resolutions (see NeuralGridModel.js's computeGridLevels).
function computeGridLatentTexels( resolutions: number[] ): number {

	return resolutions.reduce( ( sum, resolution ) => sum + resolution * resolution, 0 );

}

interface MLPLayoutEntry {
	rows: number;
	cols: number;
	kind: 'weight' | 'bias';
}

// Parameter count + FLOPs directly from a compact manifest's `mlp.layout`
// array (see NeuralBinaryCodec.js's encodeMLPLayersBase64: alternating
// `{ rows, cols, kind: 'weight' }` / `{ rows, cols, kind: 'bias' }` entries
// per layer) - lets a loaded .neuralTexture/.neuralMaterial/.neuralAppearance
// manifest report its *actual* size/cost without re-decoding every float16
// weight first.
function computeMLPLayoutStats( layout: MLPLayoutEntry[] ): { paramCount: number; flops: number } {

	let paramCount = 0;
	let flops = 0;

	for ( let i = 0; i < layout.length; i += 2 ) {

		const weightEntry = layout[ i ];
		paramCount += computeLinearParamCount( weightEntry.rows, weightEntry.cols );
		flops += computeLinearFlops( weightEntry.rows, weightEntry.cols );

	}

	return { paramCount, flops };

}

// Rolls up a model's total grid-latent scalar count (texels * channels,
// summed across every level) and total MLP parameter count/FLOPs (summed
// across every head) into the two byte footprints described above, plus the
// FLOPs passed through unchanged. Packed counts assume one stream or byte-aligned
// grid boundaries (as with the website's four-channel grids).
function computeModelFootprint( { gridParams, mlpParams, flops, gridStorageBits = 8 }: { gridParams: number; mlpParams: number; flops: number; gridStorageBits?: 2 | 4 | 8 } ): { memoryBytes: number; storageBytes: number; flops: number } {

	return {
		memoryBytes: gridParams * GRID_MEMORY_BYTES_PER_CHANNEL + mlpParams * MLP_MEMORY_BYTES_PER_PARAM,
		storageBytes: Math.ceil( gridParams * gridStorageBits / 8 ) + mlpParams * MLP_STORAGE_BYTES_PER_PARAM,
		flops
	};

}

// Builds the "Memory Size: ... Storage Size: ... N FLOP/Texel" summary line
// shared across the three examples' "network size" panels.
function formatModelSizeSummary( { memoryBytes, storageBytes, flops }: { memoryBytes: number; storageBytes: number; flops: number } ): string {

	return `Memory Size: ${formatBytes( memoryBytes )}   Storage Size: ${formatBytes( storageBytes )}   ${formatFlops( flops )}/Texel`;

}

export {
	GRID_MEMORY_BYTES_PER_CHANNEL,
	GRID_STORAGE_BYTES_PER_CHANNEL,
	MLP_MEMORY_BYTES_PER_PARAM,
	MLP_STORAGE_BYTES_PER_PARAM,
	formatBytes,
	formatFlops,
	computeLinearParamCount,
	computeLinearFlops,
	computeMLPParamCount,
	computeMLPFlops,
	computeGridLatentTexels,
	computeMLPLayoutStats,
	computeModelFootprint,
	formatModelSizeSummary
};
