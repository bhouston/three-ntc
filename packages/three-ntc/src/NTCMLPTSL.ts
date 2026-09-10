import * as THREE from 'three';
import * as TSL from 'three/tsl';

/**
 * TSL "hardGELU" - see NTCMLP.js's hardGELU doc comment for the exact
 * piecewise formula and rationale (the NVIDIA neural texture compression
 * paper's cheap GELU approximation).
 *
 * Deliberately branch-free arithmetic (`clamp`/`max`) rather than
 * `select()`, unlike NTCGPUKernelsTSL.js's scalar `hardGeluTSL` twin: this
 * one runs on the vec4-packed pre-activations evaluateLinearLayerMat4
 * evaluates 4 neurons at a time, and three.js's `select(cond, a, b)`
 * (ConditionalNode) always narrows `cond` to a single scalar `bool` before
 * branching - even when `cond` was itself a per-component `bvec4` comparison
 * - so it picks one branch for the *entire* vector rather than selecting
 * component-wise. With a per-component boolean condition (as
 * `x.greaterThanEqual(1.5)`/`x.lessThanEqual(-1.5)` on a vec4 produce), that
 * silently mis-selects whichever of the 4 packed neurons don't agree with
 * whatever the narrowed condition happened to resolve to - invisible for
 * small/random weights that never push any lane's pre-activation past the
 * +-1.5 breakpoints, but badly wrong once real training pushes some (not
 * all) lanes in a packed vec4 group past them, e.g. to represent sharp,
 * high-contrast detail. (The scalar version below has no such problem -
 * with a single float `x`, `cond` is already a scalar bool.)
 *
 * Equivalent closed form, using only per-component clamp/max (both
 * genuinely component-wise for vector types, unlike select()):
 * `hardGELU(x) = middle(clamp(x, -1.5, 1.5)) + max(x - 1.5, 0)`, where
 * `middle(t) = t/3 * (t + 1.5)` is the same quadratic middle branch as
 * before. Check each region: for `x <= -1.5`, `clamp(x,-1.5,1.5) = -1.5` so
 * `middle(-1.5) = 0`, and `max(x-1.5,0) = 0` (since `x-1.5 <= -3`) -> `0`,
 * matching the flat branch. For `x >= 1.5`, `clamp(x,-1.5,1.5) = 1.5` so
 * `middle(1.5) = 1.5`, and `max(x-1.5,0) = x-1.5` -> `1.5 + (x-1.5) = x`,
 * matching the identity branch. For `-1.5 < x < 1.5`, `clamp` is a no-op and
 * `max(x-1.5,0) = 0` (since `x < 1.5`) -> `middle(x)`, matching the middle
 * branch - continuous at both breakpoints by construction.
 */
function hardGeluTSL( x: any ): any {

	const clamped = x.clamp( - 1.5, 1.5 );
	const middle = clamped.mul( clamped.add( 1.5 ) ).div( 3 );
	const linearTail = x.sub( 1.5 ).max( 0 );

	return middle.add( linearTail );

}

function checkPackedLength( count: number, next: unknown[] ): void {

	if ( next.length !== count ) {

		throw new Error( `THREE.NTCMLPTSL: Packed buffer length mismatch (${ count } !== ${ next.length }).` );

	}

}

function copyMat4ArrayInto( targetArray: any[], matrices: any[] ): void {

	checkPackedLength( targetArray.length, matrices );

	for ( let i = 0; i < matrices.length; i ++ ) targetArray[ i ].copy( matrices[ i ] );

}

function copyVec4ArrayInto( targetArray: any[], vectors: any[] ): void {

	checkPackedLength( targetArray.length, vectors );

	for ( let i = 0; i < vectors.length; i ++ ) targetArray[ i ].copy( vectors[ i ] );

}

/** A `{ node, update }` bundle wrapping a packed `uniformArray`. */
export interface PackedStorage<T> {
	node: any;
	update: ( next: T[] ) => void;
}

/**
 * Builds the storage for one packed `mat4`-per-block MLP weight array
 * (see packLayerWeightsMat4), as a `{ node, update(matrices) }`
 * bundle: `node.element(i)` reads block `i` exactly like a plain
 * `uniformArray(..., 'mat4')` and `update(matrices)` re-uploads new weights
 * in place (used by live training-preview hot-swaps). This intentionally
 * stays on stock fp32 TSL uniforms.
 */
function createMat4Storage( matrices: any[] ): PackedStorage<any> {

	const node = TSL.uniformArray( matrices, 'mat4' );

	return { node, update: ( next: any[] ) => copyMat4ArrayInto( node.array, next ) };

}

/**
 * Same as createMat4Storage, for a packed `vec4`-per-block array (see
 * packLayerBiasesVec4) - typically an MLP layer's biases.
 */
function createVec4Storage( vectors: any[] ): PackedStorage<any> {

	const node = TSL.uniformArray( vectors, 'vec4' );

	return { node, update: ( next: any[] ) => copyVec4ArrayInto( node.array, next ) };

}

/**
 * Shared TSL-side vec4-packed MLP evaluation, used by both neural-appearance
 * (NeuralAppearanceTSL.js) and neural-texture/neural-material
 * (NeuralTextureNodeMaterial.js's evaluateNeuralTextureRaw). Each linear
 * layer's weights are packed into `mat4` uniform blocks (4 outputs x 4
 * inputs per block) and evaluated with a native `mat4 * vec4` multiply -
 * one hardware FMA-chain instruction per input quad, instead of 4 separate
 * `dot(vec4, vec4)` calls (one per output neuron). Same total FLOP count,
 * but far fewer instructions and better register/ALU utilization,
 * particularly on tile-based mobile GPUs (Apple Silicon, Adreno) with
 * dedicated 4-wide vector FMA units.
 *
 * These two module families used to have separate, independently
 * hand-written evaluators (neural-texture's was scalar-`float`-uniformArray
 * based, not vec4-packed) with no principled reason for the divergence -
 * which meant a correctness/perf fix discovered in one (the `.toVar()`
 * materialization below - see evaluateLinearLayerMat4's comment) never
 * reached the other. Consolidated here so future changes to "how do we
 * safely evaluate an MLP layer in TSL" only have to happen once.
 */

// Packs a flat array of scalar TSL nodes/plain numbers into vec4-grouped TSL
// nodes, zero-padding the final group if `inputs.length` isn't a multiple of 4.
function packVec4Inputs( inputs: ( any | number )[] ): any[] {

	const groups = [];
	const groupCount = Math.ceil( inputs.length / 4 );

	for ( let i = 0; i < groupCount; i ++ ) {

		const offset = i * 4;

		groups.push( TSL.vec4(
			inputs[ offset ] ?? 0,
			inputs[ offset + 1 ] ?? 0,
			inputs[ offset + 2 ] ?? 0,
			inputs[ offset + 3 ] ?? 0
		) );

	}

	return groups;

}

// Inverse of packVec4Inputs: extracts `outputSize` scalar nodes back out of
// a vec4-grouped array.
function unpackVec4Outputs( groups: any[], outputSize: number ): any[] {

	const outputs = [];

	for ( let i = 0; i < outputSize; i ++ ) {

		outputs.push( groups[ Math.floor( i / 4 ) ].element( i % 4 ) );

	}

	return outputs;

}

// CPU-side: packs a flat, row-major `weights[outputSize][inputSize]` array
// into THREE.Matrix4 blocks ready for a `uniformArray( ..., 'mat4' )`, one
// block per (outputVector, inputVector) quad-pair, laid out as
// `outputVector * inputVectorCount + inputVector` (matching
// evaluateLinearLayerMat4's `getWeightMat4` indexing below). Each block's
// row r holds the weights feeding output `outputVector * 4 + r` from inputs
// `inputVector * 4 .. inputVector * 4 + 3`, zero-padded past
// `inputSize`/`outputSize`. THREE.Matrix4.set() takes its 16 arguments in
// row-major order but stores them column-major internally, so this reads
// naturally as [output][input] here and `mat4Uniform.mul(vec4Input)` on the
// GPU produces the correct matrix-vector product with no transpose
// bookkeeping required at either end.
function packLayerWeightsMat4( weights: Float32Array | number[], inputSize: number, outputSize: number ): any[] {

	const inputVectorCount = Math.ceil( inputSize / 4 );
	const outputVectorCount = Math.ceil( outputSize / 4 );
	const packed: any[] = [];

	const weightAt = ( outputIndex: number, inputIndex: number ): number => {

		if ( outputIndex >= outputSize || inputIndex >= inputSize ) return 0;

		return weights[ outputIndex * inputSize + inputIndex ] || 0;

	};

	for ( let outputVector = 0; outputVector < outputVectorCount; outputVector ++ ) {

		const outputBase = outputVector * 4;

		for ( let inputVector = 0; inputVector < inputVectorCount; inputVector ++ ) {

			const inputBase = inputVector * 4;

			const matrix = new THREE.Matrix4();
			matrix.set(
				weightAt( outputBase, inputBase ), weightAt( outputBase, inputBase + 1 ), weightAt( outputBase, inputBase + 2 ), weightAt( outputBase, inputBase + 3 ),
				weightAt( outputBase + 1, inputBase ), weightAt( outputBase + 1, inputBase + 1 ), weightAt( outputBase + 1, inputBase + 2 ), weightAt( outputBase + 1, inputBase + 3 ),
				weightAt( outputBase + 2, inputBase ), weightAt( outputBase + 2, inputBase + 1 ), weightAt( outputBase + 2, inputBase + 2 ), weightAt( outputBase + 2, inputBase + 3 ),
				weightAt( outputBase + 3, inputBase ), weightAt( outputBase + 3, inputBase + 1 ), weightAt( outputBase + 3, inputBase + 2 ), weightAt( outputBase + 3, inputBase + 3 )
			);

			packed.push( matrix );

		}

	}

	return packed;

}

// CPU-side: packs a flat bias array into THREE.Vector4s, zero-padded past
// `biases.length`.
function packLayerBiasesVec4( biases: Float32Array | number[] ): any[] {

	const packed: any[] = [];
	const vectorCount = Math.ceil( biases.length / 4 );

	for ( let vectorIndex = 0; vectorIndex < vectorCount; vectorIndex ++ ) {

		const offset = vectorIndex * 4;

		packed.push( new THREE.Vector4(
			biases[ offset ] || 0,
			biases[ offset + 1 ] || 0,
			biases[ offset + 2 ] || 0,
			biases[ offset + 3 ] || 0
		) );

	}

	return packed;

}

// Evaluate a dense layer as runtime loops over mat4/vec4 blocks. The callbacks
// receive TSL integer nodes, not JavaScript indices. Keep the bounds in uniforms:
// a statically expanded MLP inside the eight-tap filter caused severe first-use
// driver stalls. Materializing layer outputs alone did not fix that expansion.
// See tsl_performance_regression_cause_and_fix.md for measured alternatives.
//
// Distinct loop names are required: nested Loop(n) calls both default to `i`,
// shadowing the outer index. The inline Fn registers assignments even when the
// raw decoder is constructed outside another Fn; it is not a WGSL function.
function evaluateLinearLayerMat4(
	inputs: any[],
	inputSize: number,
	outputSize: number,
	activation: string | undefined,
	getWeightMat4: ( outputVector: any, inputVector: any ) => any,
	getBiasVec4: ( ( outputVector: any ) => any ) | null
): any[] {

	const inputVectorCount = Math.ceil(inputSize / 4);
	const outputVectorCount = Math.ceil(outputSize / 4);
	// Small layers need at most sixteen mat4 products. Keep their indices static
	// so drivers can keep activations in registers instead of indexed arrays.
	// The shipped 8-wide brick is faster with this path; large layers retain
	// compact loops to avoid the measured compilation cliff.
	if (inputVectorCount <= 4 && outputVectorCount <= 4) {
		return Array.from({length:outputVectorCount}, (_, o) => {
			let value=getBiasVec4 ? getBiasVec4(TSL.int(o)) : TSL.vec4(0);
			for(let i=0;i<inputVectorCount;i++) value=value.add(getWeightMat4(TSL.int(o), TSL.int(i)).mul(inputs[i]));
			return (activation === 'relu' ? value.max(0) : activation === 'hgelu' ? hardGeluTSL(value) : value).toVar();
		});
	}

	const evaluated = TSL.Fn(() => {
		const packedInputs = TSL.array(inputs).toVar();
		const packedOutputs = TSL.array('vec4', outputVectorCount).toVar();
		const inputCount = TSL.uniform(inputVectorCount, 'int');
		const outputCount = TSL.uniform(outputVectorCount, 'int');
		TSL.Loop({end:outputCount, name:'ntcOutput'}, ({ntcOutput:outputVector}: {ntcOutput:any}) => {
			const value = (getBiasVec4 ? getBiasVec4(outputVector) : TSL.vec4(0)).toVar();
			TSL.Loop({end:inputCount, name:'ntcInput'}, ({ntcInput:inputVector}: {ntcInput:any}) => {
				value.addAssign(getWeightMat4(outputVector, inputVector).mul(packedInputs.element(inputVector)));
			});
			packedOutputs.element(outputVector).assign(activation === 'relu' ? value.max(0) : activation === 'hgelu' ? hardGeluTSL(value) : value);
		});
		return packedOutputs;
	})();
	const outputs = Array.from({length:outputVectorCount}, (_, i) => evaluated.element(i));

	return outputs;

}

export {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	createMat4Storage,
	createVec4Storage,
	hardGeluTSL
};
