import { FileLoader, Loader } from 'three';
import { FORMAT, VERSION, getChannel, layoutChannels, decodeUvTransform } from './NTCFormat.js';
import { LATENT_CODECS, decodeMLPLayersBase64, MLPBlock, LatentDtype } from './NTCBinaryCodec.js';
import { NTCCpuModel } from './NTCDecoderTSL.js';
import { NTCChannelClassification } from './NTCNodeMaterial.js';

/** One entry of `manifest.latents.levels` - a quantized, base64-packed feature grid. */
export interface NTCManifestLevel {
	width: number;
	height: number;
	channels: number;
	dtype: LatentDtype;
	min: number;
	max: number;
	dataBase64: string;
}

/** The `.ntc` file's top-level JSON shape - what `NTCLoader.parse()` consumes. */
export interface NTCManifest {
	format: string;
	version: number;
	name?: string;
	latents: {
		levels: NTCManifestLevel[];
		channelsPerLevel: number;
		mipsPerLevel: number;
		maxLod: number;
		wrap?: string;
		positionalEncoding?: boolean;
		dualGrid?: boolean;
	};
	mlp: MLPBlock;
	outputChannels?: number;
	uvTransform?: number[];
	channels: {
		activeKeys: string[];
		constantValues?: Record<string, any>;
	};
	renderFlags?: { side?: number; transparent?: boolean } | null;
}

/** What `NTCLoader.parse()` returns - ready for `new NTCNodeMaterial(cpuModel, channelClassification)`. */
export interface NTCParsedAsset {
	name: string;
	cpuModel: NTCCpuModel;
	channelClassification: NTCChannelClassification;
}

/**
 * A loader for `.ntc` (Neural Texture Compression) assets - a compact,
 * quantized multiresolution latent grid pyramid plus a float16-packed MLP
 * decoder, jointly fit against a set of PBR channels (see NTCManifest.js /
 * NTCFormat.js).
 *
 * `parse()` reconstructs `{ name, cpuModel, channelClassification }` -
 * exactly what `new NTCNodeMaterial( cpuModel, channelClassification,
 * options )` expects (see NTCNodeMaterial.js) - the same shape a live
 * `NTCTrainer` run / `NTCFit.fitNTCMaterial()` call produces.
 *
 * @augments Loader
 * @three_import import { NTCLoader } from 'three/addons/loaders/NTCLoader.js';
 */
class NTCLoader extends Loader {

	/**
	 * Constructs a new NTC loader.
	 *
	 * @param {LoadingManager} [manager] - The loading manager.
	 */
	constructor( manager?: any ) {

		super( manager );

	}

	/**
	 * Starts loading from the given URL and passes the parsed NTC asset to
	 * the `onLoad()` callback.
	 *
	 * @param {string} url - The path/URL of the JSON file to load.
	 * @param {function(Object)} onLoad - Executed when loading has finished.
	 * @param {onProgressCallback} onProgress - Executed while loading progresses.
	 * @param {onErrorCallback} onError - Executed when errors occur.
	 */
	load(
		url: string,
		onLoad?: ( asset: NTCParsedAsset ) => void,
		onProgress?: ( event: ProgressEvent ) => void,
		onError?: ( err: unknown ) => void
	): void {

		const scope = this;

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setResponseType( 'json' );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, function ( json: unknown ) {

			try {

				const asset = scope.parse( json as NTCManifest | string );
				if ( onLoad ) onLoad( asset );

			} catch ( e ) {

				if ( onError ) {

					onError( e );

				} else {

					console.error( e );

				}

				scope.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	/**
	 * Parses a `.ntc` manifest.
	 *
	 * @param {(Object|string)} data - The JSON manifest, either parsed or as a string.
	 * @return {Object} `{ name, cpuModel, channelClassification }`, ready for `new NTCNodeMaterial()`.
	 */
	parse( data: NTCManifest | string ): NTCParsedAsset {

		const manifest: NTCManifest = ( typeof data === 'string' ) ? JSON.parse( data ) : data;

		validateManifest( manifest );

		const grids = manifest.latents.levels.map( ( level, index ) => decodeLevel( level, `latents.levels[${ index }]` ) );
		const decoderLayers = decodeMLPLayersBase64( manifest.mlp );

		const cpuModel: NTCCpuModel = {
			channels: manifest.latents.channelsPerLevel,
			levels: grids.length,
			mipsPerLevel: manifest.latents.mipsPerLevel,
			maxLod: manifest.latents.maxLod,
			grids,
			decoder: { layers: decoderLayers },
			outputChannels: manifest.outputChannels !== undefined ?
				manifest.outputChannels : decoderLayers[ decoderLayers.length - 1 ].outputSize,
			wrap: manifest.latents.wrap || 'repeat',
			// Defaults to identity when absent (manifests saved before this
			// field existed, or one that never had a detected/explicit UV
			// transform) - see NTCFormat.js's decodeUvTransform and
			// NTCNodeMaterial.js's query-time coord build.
			uvTransform: decodeUvTransform( manifest.uvTransform ),
			// Optional/additive, defaults to false for manifests saved before
			// this field existed - see NTCDecoderTSL.js / NTCNodeMaterial.js.
			positionalEncoding: manifest.latents.positionalEncoding === true,
			dualGrid: manifest.latents.dualGrid === true
		};

		const channelClassification = decodeChannelClassification( manifest.channels, manifest.renderFlags );

		return { name: manifest.name || '', cpuModel, channelClassification };

	}

}

function decodeChannelClassification( channels: NTCManifest[ 'channels' ], renderFlags: NTCManifest[ 'renderFlags' ] ): NTCChannelClassification {

	const activeList = channels.activeKeys.map( ( key ) => getChannel( key ) );
	const { channels: activeChannels, totalChannels, packCount } = layoutChannels( activeList );

	return { activeChannels, totalChannels, packCount, constantValues: channels.constantValues || {}, renderFlags: renderFlags || null };

}

function decodeLevel( level: NTCManifestLevel, path: string ) {

	assertInteger( level.width, `${ path }.width`, 1 );
	assertInteger( level.height, `${ path }.height`, 1 );
	assertInteger( level.channels, `${ path }.channels`, 1, 4 );

	const codec = LATENT_CODECS[ level.dtype ];

	if ( codec === undefined ) {

		throw new Error( `THREE.NTCLoader: Unsupported ${ path }.dtype "${ level.dtype }".` );

	}

	const expectedLength = level.width * level.height * level.channels;
	const data = codec.decode( level.dataBase64, level.min, level.max, expectedLength );

	return { width: level.width, height: level.height, channels: level.channels, data };

}

function validateManifest( manifest: NTCManifest ): void {

	if ( manifest === null || typeof manifest !== 'object' ) {

		throw new Error( 'THREE.NTCLoader: Manifest must be an object.' );

	}

	if ( manifest.format !== FORMAT ) {

		throw new Error( `THREE.NTCLoader: Unsupported format "${ manifest.format }" (expected "${ FORMAT }").` );

	}

	if ( manifest.version !== VERSION ) {

		throw new Error( `THREE.NTCLoader: Unsupported version ${ manifest.version } (expected ${ VERSION }).` );

	}

	if ( ! manifest.latents || ! Array.isArray( manifest.latents.levels ) || manifest.latents.levels.length === 0 ) {

		throw new Error( 'THREE.NTCLoader: Manifest must define a non-empty latents.levels array.' );

	}

	// Required since VERSION 2 (see NTCFormat.js) - without these a loaded
	// model's LOD input (NTCDecoderTSL.js) can't be correctly reconstructed
	// at all: `mipsPerLevel` (with `latents.levels.length`) determines which
	// stored grid a given LOD selects, and `maxLod` is the physical mip range
	// the decoder's LOD input was normalized against.
	assertInteger( manifest.latents.mipsPerLevel, 'latents.mipsPerLevel', 1 );
	assertInteger( manifest.latents.maxLod, 'latents.maxLod', 1 );

	if ( ! manifest.mlp || typeof manifest.mlp.dataBase64 !== 'string' || ! Array.isArray( manifest.mlp.layout ) ) {

		throw new Error( 'THREE.NTCLoader: Manifest must define mlp.layout and mlp.dataBase64.' );

	}

	if ( ! manifest.channels || ! Array.isArray( manifest.channels.activeKeys ) ) {

		throw new Error( 'THREE.NTCLoader: Manifest must define channels.activeKeys.' );

	}

	for ( const key of manifest.channels.activeKeys ) {

		getChannel( key ); // throws a clear error on an unknown channel key

	}

	// Optional/additive field (see NTCFormat.js's isIdentityUvTransform) -
	// only shape-checked when present at all.
	if ( manifest.uvTransform !== undefined &&
		( ! Array.isArray( manifest.uvTransform ) || manifest.uvTransform.length !== 6 ) ) {

		throw new Error( 'THREE.NTCLoader: uvTransform must be a 6-element array when present.' );

	}

}

function assertInteger( value: number, path: string, min: number, max = Infinity ): void {

	if ( Number.isInteger( value ) === false || value < min || value > max ) {

		throw new Error( `THREE.NTCLoader: ${ path } must be an integer in [${ min }, ${ max }].` );

	}

}

export { NTCLoader };
