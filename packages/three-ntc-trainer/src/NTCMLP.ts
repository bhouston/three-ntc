const LINEAR_SCALAR_OUTPUT_HE_SCALE = 0.2;
const LINEAR_RGB_OUTPUT_HE_SCALE = 0.45;
const LINEAR_RGB_OUTPUT_BIAS = 0.3;

interface MLPLayer {
	inputSize: number;
	outputSize: number;
	weights: number[];
	biases: number[];
	activation: string;
}

interface MLP {
	layers: MLPLayer[];
}

function createMLP( inputSize: number, hiddenLayers: number[], outputSize: number, random: () => number, hiddenActivation = 'relu', outputActivation = 'linear' ): MLP {

	const sizes = [ inputSize, ...hiddenLayers, outputSize ];
	const layers: MLPLayer[] = [];

	for ( let i = 0; i < sizes.length - 1; i ++ ) {

		const input = sizes[ i ];
		const output = sizes[ i + 1 ];
		const isOutputLayer = i === sizes.length - 2;
		const activation = isOutputLayer ? outputActivation : hiddenActivation;
		const isLinearOutput = isOutputLayer && activation === 'linear';
		const isLinearRgb = isLinearOutput && output === 3;
		const scale = Math.sqrt( 2 / input ) * ( isLinearRgb ? LINEAR_RGB_OUTPUT_HE_SCALE : ( isLinearOutput ? LINEAR_SCALAR_OUTPUT_HE_SCALE : 1 ) );
		const weights = new Array( input * output );
		const biases = new Array( output ).fill( isLinearRgb ? LINEAR_RGB_OUTPUT_BIAS : 0 );

		for ( let j = 0; j < weights.length; j ++ ) {

			weights[ j ] = ( random() * 2 - 1 ) * scale;

		}

		layers.push( {
			inputSize: input,
			outputSize: output,
			weights,
			biases,
			activation
		} );

	}

	return { layers };

}

function forwardMLP( mlp: MLP, input: number[] ) {

	const activations = [ input.slice() ];
	const preActivations: number[][] = [];
	let values = input.slice();

	for ( const layer of mlp.layers ) {

		const next = new Array( layer.outputSize );
		const pre = new Array( layer.outputSize );

		for ( let output = 0; output < layer.outputSize; output ++ ) {

			let value = layer.biases[ output ];

			for ( let inputIndex = 0; inputIndex < layer.inputSize; inputIndex ++ ) {

				value += layer.weights[ output * layer.inputSize + inputIndex ] * values[ inputIndex ];

			}

			pre[ output ] = value;
			next[ output ] = activate( value, layer.activation );

		}

		preActivations.push( pre );
		activations.push( next );
		values = next;

	}

	return { activations, preActivations, output: values };

}

function activate( value: number, activation: string ): number {

	if ( activation === 'relu' ) return Math.max( 0, value );
	if ( activation === 'leakyRelu' ) return value >= 0 ? value : value * 0.01;
	if ( activation === 'tanh' ) return Math.tanh( value );
	if ( activation === 'hgelu' ) return hardGELU( value );

	return value;

}

/**
 * "hardGELU" - a cheap 3-piece approximation of GELU used by the NVIDIA
 * neural texture compression paper (Section 4.4) in place of the true
 * (erf-based) GELU, similar in shape to hard-Swish:
 *
 *   hardGELU(x) = 0                if x <= -1.5
 *               = x                if x >= 1.5
 *               = x/3 * (x + 1.5)  otherwise
 *
 * Continuous (C0) at both breakpoints - the middle branch evaluates to
 * exactly 0 at x = -1.5 and to exactly 1.5 (= x) at x = 1.5, matching the
 * outer branches' values there - but not differentiable (C1) at either
 * breakpoint: the middle branch's own slope at x = -1.5 is -0.5, not the
 * outer branch's 0, and at x = 1.5 is 1.5, not the outer branch's 1 (see
 * hardGELUDerivative below, and compare ReLU's own single non-differentiable
 * point at x = 0). The boundaries are written as `<=`/`>=` (closed on the
 * outer branches) purely so both breakpoints evaluate through the flat/
 * identity branch directly rather than the algebraically-equal middle-branch
 * expression, which can otherwise land on IEEE-754 negative zero at x = -1.5
 * (`-1.5/3 * 0 === -0`, not `0`).
 */
function hardGELU( value: number ): number {

	if ( value <= - 1.5 ) return 0;
	if ( value >= 1.5 ) return value;

	return value / 3 * ( value + 1.5 );

}

/**
 * Derivative of hardGELU, for the hand-differentiated backward pass (see
 * NTCGPUKernelsTSL.js's hardGeluDerivativeTSL, its GPU-side counterpart).
 * Since hardGELU isn't differentiable at its two breakpoints (see its doc
 * comment), the value returned there is a boundary convention - the outer
 * (flat/identity) branch's own derivative, exactly like ReLU conventionally
 * returning 0 (not 1) as its own derivative at x = 0:
 *
 *   hardGELU'(x) = 0            if x <= -1.5
 *                = 1            if x >= 1.5
 *                = (2x + 1.5)/3 otherwise
 */
function hardGELUDerivative( value: number ): number {

	if ( value <= - 1.5 ) return 0;
	if ( value >= 1.5 ) return 1;

	return ( 2 * value + 1.5 ) / 3;

}

function sigmoid( value: number ): number {

	return 1 / ( 1 + Math.exp( - value ) );

}

function powerLog( value: number, power: number ): number {

	return power * ( Math.pow( value, 1 / power ) - 1 );

}

export {
	createMLP,
	forwardMLP,
	activate,
	hardGELU,
	hardGELUDerivative,
	sigmoid,
	powerLog
};
export type { MLP, MLPLayer };
