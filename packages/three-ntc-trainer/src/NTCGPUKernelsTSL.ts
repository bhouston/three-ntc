import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, Loop, atomicAdd, atomicLoad, atomicStore, float, instanceIndex, int, max, min, pow, select, sqrt, storage } from 'three/tsl';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from './NTCGPUTrainingConstants.js';

// Small WebGPU compute-kernel TSL builders shared by every neural trainer
// (texture, material, appearance): wrapping a grid index, clipping by a
// global gradient norm, the two zero-fill kernels that reset gradient
// accumulators between optimizer passes, one dense layer's forward/backward
// math, and the Adam step itself. None of this depends on what the
// gradients are gradients *of*, only on the fixed-point atomic layout every
// trainer shares via `NeuralGPUTrainingConstants.js`.

// TSL node graphs are dynamically typed at build time - `any` here (and
// throughout this file) stands in for the TSL `Node` type rather than
// over-specifying it.
type TSLNode = any;

function wrapIndexTSL( val: TSLNode, size: number ): TSLNode {

	return val.mod( size ).add( size ).mod( size );

}

/**
 * Scalar TSL "hardGELU" - see NTCMLP.js's hardGELU doc comment for the exact
 * piecewise formula (the NVIDIA neural texture compression paper's cheap
 * GELU approximation) and NTCMLPTSL.js's hardGeluTSL for the vec4-flavored
 * twin used at inference; this one operates on the plain scalar `float`
 * activations the hand-written training kernels below read/write from
 * storage buffers.
 */
function hardGeluTSL( x: TSLNode ): TSLNode {

	const middle = x.mul( x.add( 1.5 ) ).div( 3 );
	const upper = select( x.greaterThanEqual( 1.5 ), x, middle );

	return select( x.lessThanEqual( - 1.5 ), float( 0.0 ), upper );

}

/**
 * Derivative of hardGeluTSL, for the hand-differentiated backward pass - see
 * NTCMLP.js's hardGELUDerivative for the plain-JS reference (including its
 * doc comment on the boundary convention at the two non-differentiable
 * breakpoints) this must match.
 */
function hardGeluDerivativeTSL( x: TSLNode ): TSLNode {

	const middle = x.mul( 2.0 / 3.0 ).add( 0.5 );
	const upper = select( x.greaterThanEqual( 1.5 ), float( 1.0 ), middle );

	return select( x.lessThanEqual( - 1.5 ), float( 0.0 ), upper );

}

/**
 * Applies a hidden-layer activation (forward direction) to a scalar
 * pre-activation value `z`. `'relu'` (the default) and `'hgelu'` are the only
 * two hidden activations any NTC trainer currently builds (see
 * NTCGridPyramidModel.js's `hiddenActivation` option) - an unrecognized name
 * falls back to `relu` rather than throwing, matching NTCMLP.js's `activate`
 * being permissive about unknown names.
 */
function applyHiddenActivationTSL( z: TSLNode, activation: string ): TSLNode {

	if ( activation === 'hgelu' ) return hardGeluTSL( z );

	return max( z, float( 0.0 ) );

}

/**
 * Derivative of applyHiddenActivationTSL at pre-activation value `z` (i.e.
 * gate a backward delta by this to propagate it through the same activation
 * applyHiddenActivationTSL applied going forward).
 */
function activationDerivativeTSL( z: TSLNode, activation: string ): TSLNode {

	if ( activation === 'hgelu' ) return hardGeluDerivativeTSL( z );

	return select( z.greaterThan( 0.0 ), float( 1.0 ), float( 0.0 ) );

}

interface AdamParameterBuffers {
	attribute: InstanceType<typeof StorageBufferAttribute>;
	gradAttribute: InstanceType<typeof StorageBufferAttribute>;
	mAttribute: InstanceType<typeof StorageBufferAttribute>;
	vAttribute: InstanceType<typeof StorageBufferAttribute>;
	valuesStorage: TSLNode;
	gradAtomic: TSLNode;
	mStorage: TSLNode;
	vStorage: TSLNode;
}

/**
 * Allocates the 4 GPU StorageBuffers one Adam-optimized parameter block
 * needs (value, atomic gradient accumulator, and the 1st/2nd moment
 * buffers Adam maintains per parameter) plus their `storage()`-wrapped TSL
 * nodes, as a single bundle object - `{ attribute, gradAttribute,
 * mAttribute, vAttribute, valuesStorage, gradAtomic, mStorage, vStorage }`.
 * Every neural trainer's GPUModel (texture, appearance) calls this once for
 * its weights and once for its latents (`this.weightsBuffers =
 * createAdamParameterBuffers(totalWeights)`, `this.latentsBuffers =
 * createAdamParameterBuffers(totalLatents)`) - identical buffer shape in
 * both, just sized differently - so this constructor boilerplate, and the
 * "a set of Adam parameter buffers" concept it represents, lives here
 * instead of being hand-duplicated (or hand-flattened into 8 loose fields)
 * per trainer.
 */
function createAdamParameterBuffers( count: number ): AdamParameterBuffers {

	const attribute = new StorageBufferAttribute( new Float32Array( count ), 1, Float32Array );
	const gradAttribute = new StorageBufferAttribute( new Int32Array( count ), 1, Int32Array );
	const mAttribute = new StorageBufferAttribute( new Float32Array( count ), 1, Float32Array );
	const vAttribute = new StorageBufferAttribute( new Float32Array( count ), 1, Float32Array );

	return {
		attribute,
		gradAttribute,
		mAttribute,
		vAttribute,
		valuesStorage: storage( attribute, 'float', count ),
		gradAtomic: storage( gradAttribute, 'int', count ).toAtomic(),
		mStorage: storage( mAttribute, 'float', count ),
		vStorage: storage( vAttribute, 'float', count )
	};

}

/**
 * Releases the 4 GPU StorageBuffers a `createAdamParameterBuffers` bundle
 * owns - the disposal counterpart kept next to it for the same reason.
 * `renderer`, when given, also releases the backend-side GPU buffer via
 * `renderer.backend.destroyAttribute` - `StorageBufferAttribute.dispose()`
 * alone only fires a 'dispose' event, and these standalone compute-only
 * buffers (never attached to a BufferGeometry/render object) have nothing
 * wired up to listen for it.
 */
function disposeAdamParameterBuffers( buffers: AdamParameterBuffers, renderer: any = null ): void {

	for ( const attribute of [ buffers.attribute, buffers.gradAttribute, buffers.mAttribute, buffers.vAttribute ] ) {

		if ( renderer && renderer.backend && renderer.backend.has( attribute ) ) renderer.backend.destroyAttribute( attribute );
		attribute.dispose();

	}

}

/**
 * Forward pass through one dense layer: `out = W . in + b`, optionally
 * activated (see applyHiddenActivationTSL - `'relu'` by default, or
 * `'hgelu'`). `inputBase`/`zBase`/`aBase` are absolute activation-buffer
 * indices (already offset by whatever per-sample/per-section offset applies)
 * rather than raw layout offsets, so this same helper covers both "this
 * layer's input starts partway through a shared activation buffer" and
 * "this layer's input is its own small scratch region" uniformly. `aBase:
 * null` skips writing the activated output - used for the final (linear)
 * layer of a head, whose raw `z` is consumed directly.
 *
 * Shared by every trainer's forward pass - neural-appearance's decoder/IBL/
 * indirect-probe heads (NeuralAppearanceGPUComputeTSL.js) and neural-
 * texture's MLP decoder (NeuralTextureGPUComputeTSL.js).
 */
function forwardDenseLayerTSL( { activationsStorage, weightsStorage, inputBase, inputSize, outputSize, weightsOffset, biasesOffset, zBase, aBase = null, activation = 'relu' }: {
	activationsStorage: TSLNode; weightsStorage: TSLNode; inputBase: TSLNode; inputSize: number; outputSize: number;
	weightsOffset: number; biasesOffset: number; zBase: TSLNode; aBase?: TSLNode | null; activation?: string;
} ): void {

	Loop( { start: 0, end: outputSize, type: 'int', name: 'j', condition: '<' }, ( { j }: { j: TSLNode } ) => {

		const val = weightsStorage.element( int( biasesOffset ).add( j ) ).toVar();
		const rowOffset = int( weightsOffset ).add( j.mul( inputSize ) );

		Loop( { start: 0, end: inputSize, type: 'int', name: 'i', condition: '<' }, ( { i }: { i: TSLNode } ) => {

			val.addAssign( weightsStorage.element( rowOffset.add( i ) ).mul( activationsStorage.element( inputBase.add( i ) ) ) );

		} );

		activationsStorage.element( zBase.add( j ) ).assign( val );
		if ( aBase !== null ) activationsStorage.element( aBase.add( j ) ).assign( applyHiddenActivationTSL( val, activation ) );

	} );

}

/**
 * Backward pass, gradient-accumulation half: for one dense layer, scatters
 * `delta (x) input` into that layer's weight/bias gradients via atomic
 * fixed-point adds. Shared by the same call sites as `forwardDenseLayerTSL`
 * above (their backward counterparts).
 */
function accumulateDenseLayerGradTSL( { activationsStorage, gradWeightsAtomic, deltaBase, inputBase, inputSize, outputSize, weightsOffset, biasesOffset }: {
	activationsStorage: TSLNode; gradWeightsAtomic: TSLNode; deltaBase: TSLNode; inputBase: TSLNode; inputSize: number;
	outputSize: number; weightsOffset: number; biasesOffset: number;
} ): void {

	Loop( { start: 0, end: outputSize, type: 'int', name: 'j', condition: '<' }, ( { j }: { j: TSLNode } ) => {

		const delta_j = activationsStorage.element( deltaBase.add( j ) );
		atomicAdd( gradWeightsAtomic.element( int( biasesOffset ).add( j ) ), int( delta_j.mul( float( FIXED_POINT_SCALE ) ) ) );
		const rowOffset = int( weightsOffset ).add( j.mul( inputSize ) );

		Loop( { start: 0, end: inputSize, type: 'int', name: 'i', condition: '<' }, ( { i }: { i: TSLNode } ) => {

			const in_i = activationsStorage.element( inputBase.add( i ) );
			atomicAdd( gradWeightsAtomic.element( rowOffset.add( i ) ), int( delta_j.mul( in_i ).mul( float( FIXED_POINT_SCALE ) ) ) );

		} );

	} );

}

/**
 * Backward pass, delta-propagation half: for one dense layer whose *input*
 * came from an activated hidden layer, propagates `delta` back through the
 * weight matrix and gates it by that input's own activation derivative (see
 * activationDerivativeTSL - `'relu'` by default, or `'hgelu'`), writing the
 * previous layer's delta. NOT a fit for a layer whose input is raw
 * (unactivated) features - those have no activation derivative to gate by -
 * so those stay hand-written at their call sites.
 */
function backwardDenseLayerTSL( { activationsStorage, weightsStorage, deltaBase, deltaSize, weightsOffset, prevSize, prevZBase, outDeltaBase, activation = 'relu' }: {
	activationsStorage: TSLNode; weightsStorage: TSLNode; deltaBase: TSLNode; deltaSize: number; weightsOffset: number;
	prevSize: number; prevZBase: TSLNode; outDeltaBase: TSLNode; activation?: string;
} ): void {

	Loop( { start: 0, end: prevSize, type: 'int', name: 'i', condition: '<' }, ( { i }: { i: TSLNode } ) => {

		const gradInput_i = float( 0.0 ).toVar();

		Loop( { start: 0, end: deltaSize, type: 'int', name: 'j', condition: '<' }, ( { j }: { j: TSLNode } ) => {

			const delta_j = activationsStorage.element( deltaBase.add( j ) );
			const w_ji = weightsStorage.element( int( weightsOffset ).add( j.mul( prevSize ) ).add( i ) );
			gradInput_i.addAssign( delta_j.mul( w_ji ) );

		} );

		const z_i = activationsStorage.element( prevZBase.add( i ) );
		const delta_i = gradInput_i.mul( activationDerivativeTSL( z_i, activation ) );
		activationsStorage.element( outDeltaBase.add( i ) ).assign( delta_i );

	} );

}

function computeGradientClipScale( gradNormAtomic: TSLNode, maxGradientNormUniform: TSLNode ): TSLNode {

	const normSquared = float( atomicLoad( gradNormAtomic.element( 0 ) ) ).div( float( GRADIENT_NORM_SCALE ) );
	const unclippedScale = maxGradientNormUniform.div( sqrt( max( normSquared, float( 1e-20 ) ) ) );

	return min( float( 1.0 ), unclippedScale );

}

/**
 * Clears the scalar accumulator used by the gradient clipping pass.
 */
function createResetGradientNormComputeNode( gpuModel: { gradNormAtomic: TSLNode } ): TSLNode {

	const { gradNormAtomic } = gpuModel;

	return Fn( () => {

		atomicStore( gradNormAtomic.element( 0 ), int( 0 ) );

	} )().compute( 1 ).setName( 'NTCResetGradientNorm' );

}

/**
 * Clears accumulated weight and latent gradients before a scoped optimizer pass.
 */
function createResetGradientsComputeNode( gpuModel: { layout: { totalWeights: number; totalLatents: number }; weightsBuffers: AdamParameterBuffers; latentsBuffers: AdamParameterBuffers } ): TSLNode {

	const { layout } = gpuModel;
	const { gradAtomic: gradWeightsAtomic } = gpuModel.weightsBuffers;
	const { gradAtomic: gradLatentsAtomic } = gpuModel.latentsBuffers;
	const dispatchCount = layout.totalWeights + layout.totalLatents;

	return Fn( () => {

		const idx = int( instanceIndex );

		If( idx.lessThan( int( layout.totalWeights ) ), () => {

			atomicStore( gradWeightsAtomic.element( idx ), int( 0 ) );

		} ).Else( () => {

			atomicStore( gradLatentsAtomic.element( idx.sub( int( layout.totalWeights ) ) ), int( 0 ) );

		} );

	} )().compute( dispatchCount ).setName( 'NTCResetGradients' );

}

/**
 * Builds one Adam optimizer step, applied to a single fixed-point gradient
 * buffer (weights or latents, whichever `valuesStorage`/`gradAtomic`/
 * `mStorage`/`vStorage` point at). Shared by every trainer's weight- and
 * latent-update kernels: same bias-corrected moment estimates, same
 * fixed-point gradient decode, same global-norm clip. What differs between
 * call sites is which buffers it targets, whether the gradient still needs
 * dividing down to a per-sample average (`invBatchUniform`), and whether the
 * dispatch is scoped to a sub-range of the buffer (`offset`/`count`) rather
 * than the whole thing - the appearance model's staged BRDF/IBL training
 * passes need that, texture and material always update the full buffer.
 */
function createAdamComputeNode( {
	valuesStorage,
	gradAtomic,
	mStorage,
	vStorage,
	gradNormAtomic,
	maxGradientNormUniform,
	learningRateUniform,
	stepUniform,
	invBatchUniform = null,
	offset = 0,
	count,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-7,
	name
}: {
	valuesStorage: TSLNode; gradAtomic: TSLNode; mStorage: TSLNode; vStorage: TSLNode; gradNormAtomic: TSLNode;
	maxGradientNormUniform: TSLNode; learningRateUniform: TSLNode; stepUniform: TSLNode; invBatchUniform?: TSLNode | null;
	offset?: number; count: number; beta1?: number; beta2?: number; epsilon?: number; name: string;
} ): TSLNode {

	return Fn( () => {

		const idx = int( instanceIndex ).add( int( offset ) );
		let rawGrad = float( atomicLoad( gradAtomic.element( idx ) ) ).div( float( FIXED_POINT_SCALE ) );
		if ( invBatchUniform !== null ) rawGrad = rawGrad.mul( invBatchUniform );
		const grad = rawGrad.mul( computeGradientClipScale( gradNormAtomic, maxGradientNormUniform ) );
		atomicStore( gradAtomic.element( idx ), int( 0 ) );

		const m = mStorage.element( idx );
		const v = vStorage.element( idx );
		const value = valuesStorage.element( idx );

		const nextM = float( beta1 ).mul( m ).add( float( 1.0 - beta1 ).mul( grad ) );
		const nextV = float( beta2 ).mul( v ).add( float( 1.0 - beta2 ).mul( grad ).mul( grad ) );
		mStorage.element( idx ).assign( nextM );
		vStorage.element( idx ).assign( nextV );

		const beta1Corr = float( 1.0 ).sub( pow( float( beta1 ), float( stepUniform ) ) );
		const beta2Corr = float( 1.0 ).sub( pow( float( beta2 ), float( stepUniform ) ) );
		const mHat = nextM.div( max( beta1Corr, float( 1e-10 ) ) );
		const vHat = nextV.div( max( beta2Corr, float( 1e-10 ) ) );

		const stepVal = learningRateUniform.mul( mHat ).div( sqrt( max( vHat, float( 0.0 ) ) ).add( float( epsilon ) ) );
		valuesStorage.element( idx ).assign( value.sub( stepVal ) );

	} )().compute( count ).setName( name );

}

export {
	wrapIndexTSL,
	hardGeluTSL,
	hardGeluDerivativeTSL,
	applyHiddenActivationTSL,
	activationDerivativeTSL,
	createAdamParameterBuffers,
	disposeAdamParameterBuffers,
	forwardDenseLayerTSL,
	accumulateDenseLayerGradTSL,
	backwardDenseLayerTSL,
	computeGradientClipScale,
	createResetGradientNormComputeNode,
	createResetGradientsComputeNode,
	createAdamComputeNode
};
export type { AdamParameterBuffers, TSLNode };
