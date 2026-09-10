import { DataUtils } from 'three';

// Shared base64/half-float/uint8 codec used by every neural-* binary
// manifest format (neural-texture/.neuralTexture, neural-material/
// .neuralMaterial, neural-appearance/.neuralAppearance - see
// NeuralQuantization.js for the QAT scheme these byte encodings mirror).
// Keeps the "flat float array <-> compact base64 blob" plumbing in one
// place so every manifest module encodes/decodes identically.

const CHUNK_SIZE = 8192;

/**
 * Base64-encodes raw bytes, chunked through `String.fromCharCode.apply` to
 * avoid blowing the call stack on large arrays (spreading a 10MB+ typed
 * array as individual arguments in one call throws "Maximum call stack size
 * exceeded" in every JS engine).
 */
function base64FromBytes( bytes: Uint8Array ): string {

	if ( typeof Buffer !== 'undefined' ) {

		return Buffer.from( bytes.buffer, bytes.byteOffset, bytes.byteLength ).toString( 'base64' );

	}

	let binary = '';

	for ( let i = 0; i < bytes.length; i += CHUNK_SIZE ) {

		const chunk = bytes.subarray( i, i + CHUNK_SIZE );
		binary += String.fromCharCode.apply( null, chunk as unknown as number[] );

	}

	return btoa( binary );

}

/**
 * Decodes a base64 string back to raw bytes - the inverse of
 * `base64FromBytes`.
 */
function bytesFromBase64( str: string ): Uint8Array {

	if ( typeof Buffer !== 'undefined' ) {

		const buffer = Buffer.from( str, 'base64' );
		return new Uint8Array( buffer.buffer, buffer.byteOffset, buffer.byteLength ).slice();

	}

	const binary = atob( str );
	const bytes = new Uint8Array( binary.length );

	for ( let i = 0; i < binary.length; i ++ ) {

		bytes[ i ] = binary.charCodeAt( i );

	}

	return bytes;

}

/**
 * Scalar float32 -> float16 bit conversion, reusing `DataUtils.toHalfFloat`
 * (the same core three.js already ships and NeuralHalfFloatTexture.js's
 * `packHalfFloatRGBA` already relies on) rather than re-deriving the bit
 * twiddling.
 */
function float32ToFloat16( x: number ): number {

	return DataUtils.toHalfFloat( x );

}

/**
 * Scalar float16 -> float32 bit conversion - the missing inverse of
 * `float32ToFloat16` above, needed to decode a `.neuralTexture`/
 * `.neuralMaterial`/`.neuralAppearance` MLP weight blob back into a
 * Float32Array. Reuses `DataUtils.fromHalfFloat`.
 */
function float16ToFloat32( h: number ): number {

	return DataUtils.fromHalfFloat( h );

}

/**
 * Packs a Float32Array into a base64 string of little-endian Float16 bytes
 * (2 bytes/value) - used for MLP weight/bias blobs.
 */
function encodeFloat16Base64( data: Float32Array ): string {

	const bytes = new Uint8Array( data.length * 2 );
	const view = new DataView( bytes.buffer );

	for ( let i = 0; i < data.length; i ++ ) {

		view.setUint16( i * 2, float32ToFloat16( data[ i ] ), true );

	}

	return base64FromBytes( bytes );

}

/**
 * Decodes a base64 Float16 blob (as produced by `encodeFloat16Base64`) back
 * into a Float32Array of `length` values.
 */
function decodeFloat16Base64( str: string, length: number ): Float32Array {

	const bytes = bytesFromBase64( str );
	const view = new DataView( bytes.buffer, bytes.byteOffset, bytes.byteLength );
	const out = new Float32Array( length );

	for ( let i = 0; i < length; i ++ ) {

		out[ i ] = float16ToFloat32( view.getUint16( i * 2, true ) );

	}

	return out;

}

/**
 * Quantizes a Float32Array to a base64 string of Uint8 bytes: per-element
 * `clamp((x - min) / (max - min), 0, 1)`, rounded to the nearest of 256
 * discrete levels - used for latent-grid blobs. `min === max` (a constant
 * grid) is handled by treating every value as level 0, rather than dividing
 * by zero.
 */
function encodeUint8Base64( data: Float32Array, min: number, max: number ): string {

	const bytes = new Uint8Array( data.length );
	const range = max - min;

	for ( let i = 0; i < data.length; i ++ ) {

		const t = range !== 0 ? Math.min( 1, Math.max( 0, ( data[ i ] - min ) / range ) ) : 0;
		bytes[ i ] = Math.round( t * 255 );

	}

	return base64FromBytes( bytes );

}

/**
 * Decodes a base64 Uint8 blob (as produced by `encodeUint8Base64`) back into
 * a Float32Array of `length` values, using the same `min`/`max` range.
 */
function decodeUint8Base64( str: string, min: number, max: number, length: number ): Float32Array {

	const bytes = bytesFromBase64( str );
	const out = new Float32Array( length );
	const range = max - min;

	for ( let i = 0; i < length; i ++ ) {

		out[ i ] = min + ( bytes[ i ] / 255 ) * range;

	}

	return out;

}

/**
 * Like `encodeUint8Base64` but 16 levels, packed two values per byte (low
 * nibble first); odd lengths pad the last high nibble with 0.
 */
function encodeUint4Base64( data: Float32Array, min: number, max: number ): string {

	const bytes = new Uint8Array( ( data.length + 1 ) >> 1 );
	const range = max - min;

	for ( let i = 0; i < data.length; i ++ ) {

		const t = range !== 0 ? Math.min( 1, Math.max( 0, ( data[ i ] - min ) / range ) ) : 0;
		bytes[ i >> 1 ] |= Math.round( t * 15 ) << ( ( i & 1 ) * 4 );

	}

	return base64FromBytes( bytes );

}

/** Inverse of `encodeUint4Base64`. */
function decodeUint4Base64( str: string, min: number, max: number, length: number ): Float32Array {

	const bytes = bytesFromBase64( str );
	const out = new Float32Array( length );
	const range = max - min;

	for ( let i = 0; i < length; i ++ ) {

		const level = ( bytes[ i >> 1 ] >> ( ( i & 1 ) * 4 ) ) & 15;
		out[ i ] = min + ( level / 15 ) * range;

	}

	return out;

}

/** One fully-connected layer, in `createMLP`-shape (see NeuralMLP.js). */
export interface MLPLayer {
	inputSize: number;
	outputSize: number;
	activation?: string;
	weights: Float32Array;
	biases: Float32Array;
}

/** One `{ rows, cols, kind }` entry describing a slice of the flat MLP blob. */
export interface MLPLayoutEntry {
	rows: number;
	cols: number;
	kind: 'weight' | 'bias';
	activation?: string;
}

/** The `{ dtype, layout, dataBase64 }` block `encodeMLPLayersBase64` produces. */
export interface MLPBlock {
	dtype: 'float16';
	layout: MLPLayoutEntry[];
	dataBase64: string;
}

/**
 * Flattens an array of `createMLP`-shaped layers (see NeuralMLP.js -
 * `{ inputSize, outputSize, activation, weights, biases }`) into a single
 * `{ dtype: 'float16', layout, dataBase64 }` block: one `dataBase64` blob
 * holding every layer's weights immediately followed by its biases, in
 * layer order (hidden1 -> hidden2 -> ... -> output), and a `layout` array
 * describing how to slice that flat blob back apart - one `{ rows, cols,
 * kind: 'weight' }` entry (`rows === inputSize`, `cols === outputSize`,
 * `weights.length === rows * cols`) immediately followed by one `{ rows: 1,
 * cols: outputSize, kind: 'bias' }` entry, per layer. `activation` is
 * carried on the weight entry (not a separate parallel array) purely for
 * convenience/robustness on the decode side - every neural-* trainer's
 * `createMLP` calls already fix a `hiddenActivation`/`outputActivation` pair
 * (see NeuralTextureModel.js/NeuralAppearanceModel.js), so decoders could in
 * principle re-derive it from layer position alone, but storing it removes
 * that assumption. Shared by NeuralTextureManifest.js and
 * NeuralAppearanceManifest.js so a decoder MLP's layer layout is encoded
 * identically in both `.neuralTexture`/`.neuralMaterial` and
 * `.neuralAppearance` manifests.
 */
function encodeMLPLayersBase64( layers: MLPLayer[] ): MLPBlock {

	const layout: MLPLayoutEntry[] = [];
	let totalLength = 0;

	for ( const layer of layers ) {

		totalLength += layer.inputSize * layer.outputSize + layer.outputSize;

	}

	const flat = new Float32Array( totalLength );
	let cursor = 0;

	for ( const layer of layers ) {

		layout.push( { rows: layer.inputSize, cols: layer.outputSize, kind: 'weight', activation: layer.activation || 'linear' } );

		for ( let i = 0; i < layer.weights.length; i ++ ) flat[ cursor ++ ] = layer.weights[ i ];

		layout.push( { rows: 1, cols: layer.outputSize, kind: 'bias' } );

		for ( let i = 0; i < layer.biases.length; i ++ ) flat[ cursor ++ ] = layer.biases[ i ];

	}

	return { dtype: 'float16', layout, dataBase64: encodeFloat16Base64( flat ) };

}

/**
 * Inverse of `encodeMLPLayersBase64` - reconstructs an array of
 * `createMLP`-shaped layers (`{ inputSize, outputSize, activation, weights,
 * biases }`, `weights`/`biases` as `Float32Array`s) from a `{ dtype, layout,
 * dataBase64 }` block.
 */
function decodeMLPLayersBase64( mlp: MLPBlock ): MLPLayer[] {

	let totalLength = 0;
	for ( const entry of mlp.layout ) totalLength += entry.rows * entry.cols;

	const flat = decodeFloat16Base64( mlp.dataBase64, totalLength );
	const layers: MLPLayer[] = [];
	let cursor = 0;

	for ( let i = 0; i < mlp.layout.length; i += 2 ) {

		const weightEntry = mlp.layout[ i ];
		const biasEntry = mlp.layout[ i + 1 ];
		const inputSize = weightEntry.rows;
		const outputSize = weightEntry.cols;
		const weightsCount = inputSize * outputSize;

		const weights = flat.slice( cursor, cursor + weightsCount ); cursor += weightsCount;
		const biases = flat.slice( cursor, cursor + biasEntry.cols ); cursor += biasEntry.cols;

		layers.push( { inputSize, outputSize, activation: weightEntry.activation || 'linear', weights, biases } );

	}

	return layers;

}

export {
	base64FromBytes,
	bytesFromBase64,
	float32ToFloat16,
	float16ToFloat32,
	encodeFloat16Base64,
	decodeFloat16Base64,
	encodeUint8Base64,
	decodeUint8Base64,
	encodeUint4Base64,
	decodeUint4Base64,
	encodeMLPLayersBase64,
	decodeMLPLayersBase64
};
