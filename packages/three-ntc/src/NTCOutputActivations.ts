import { exp, float, log, max, tanh } from 'three/tsl';

/**
 * Per-channel output nonlinearities for a decoder whose MLP itself always
 * ends in a plain linear layer (see ./NTCGridPyramidModel.js
 * / NTCGPUModel.js). Consumers (currently
 * ./NTCFormat.js's per-channel `activation`
 * metadata) apply these as a post-processing step on the raw per-channel
 * output `z`, both at inference time
 * (./NTCNodeMaterial.js) and inside the GPU
 * training kernel (./NTCGPUComputeTSL.js), so the
 * two stay numerically consistent. Lives in this shared `neural/` directory
 * (rather than under `neural-material/`) because the GPU training kernel is
 * itself a `neural-texture` module reused by `neural-material` - keeping it
 * here avoids a `neural-texture` -> `neural-material` dependency.
 *
 * 'linear' (the default) is intentionally not handled by name below -
 * callers that never assign a channel's `z` through `applyChannelActivation`
 * get plain linear behavior "for free", matching every other MLP output in
 * this codebase (see NeuralMLP.js's `activate()`).
 */

/** The output nonlinearity a channel is activated with - see NTCFormat.ts's `CHANNELS`. */
export type NTCActivation = 'sigmoid' | 'tanh' | 'softplus' | 'linear' | undefined;

/**
 * Plain logistic sigmoid, `1 / (1 + e^-x)` - factored out since every
 * sigmoid-shaped output nonlinearity in this codebase (channel activations
 * here, neural-appearance's roughness/opacity/scaledSigmoid outputs in
 * ../neural-appearance/NeuralAppearanceTSL.js) is this same formula.
 */
function sigmoidTSL( xNode: any ): any {

	return float( 1 ).div( float( 1 ).add( exp( xNode.negate() ) ) );

}

/**
 * z -> a, the forward nonlinearity.
 */
function applyChannelActivation( zNode: any, activation: NTCActivation ): any {

	if ( activation === 'sigmoid' ) return sigmoidTSL( zNode );
	if ( activation === 'tanh' ) return tanh( zNode );

	// Numerically stable softplus: log(1+e^z) computed as
	// max(z,0) + log(1+e^-|z|), which never overflows exp() for large |z|.
	if ( activation === 'softplus' ) return max( zNode, float( 0 ) ).add( log( exp( zNode.abs().negate() ).add( 1 ) ) );

	return zNode;

}

/**
 * da/dz, expressed in terms of the already-computed activated output `a`
 * (cheaper than recomputing exp/tanh from `z`) - used by the GPU training
 * kernel's hand-written backward pass to convert an output-space loss delta
 * (`a - target`) into the pre-activation delta (`dL/dz`) the rest of the
 * backward pass expects.
 */
function channelActivationDerivativeFromOutput( aNode: any, activation: NTCActivation ): any {

	if ( activation === 'sigmoid' ) return aNode.mul( float( 1 ).sub( aNode ) );
	if ( activation === 'tanh' ) return float( 1 ).sub( aNode.mul( aNode ) );

	// softplus: a = log(1+e^z)  =>  e^-a = 1/(1+e^z) = 1 - sigmoid(z)
	// so da/dz = sigmoid(z) = 1 - e^-a.
	if ( activation === 'softplus' ) return float( 1 ).sub( exp( aNode.negate() ) );

	return float( 1 );

}

export { sigmoidTSL, applyChannelActivation, channelActivationDerivativeFromOutput };
