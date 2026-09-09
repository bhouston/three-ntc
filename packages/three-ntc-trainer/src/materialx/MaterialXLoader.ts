import { FileLoader, Loader } from 'three/webgpu';

import { MaterialXDocument, type MaterialXParseResult } from './MaterialXDocument.js';
import { MaterialXLog } from './MaterialXLog.js';
import { isZipBuffer, readMtlxArchive, createArchiveResolver } from './MaterialXArchive.js';

const _textDecoder = new TextDecoder();

function getResourcePath( loaderPath: string, url: string ): string {

	if ( loaderPath ) return loaderPath;

	const index = url.lastIndexOf( '/' );
	return index === - 1 ? '' : url.slice( 0, index + 1 );

}

export interface MaterialXLoaderParseOptions {
	path?: string;
	materialName?: string;
	uvSpace?: 'bottom-left' | 'top-left';
	interfaceValidator?: ( rootNode: any, log: MaterialXLog ) => void;
	throwOnErrors?: boolean;
	archiveResolver?: ( ( uri: string ) => string | null ) | null;
}

/**
 * A loader for the MaterialX format.
 *
 * The node materials loaded with this loader can only be used with WebGPURenderer.
 * Besides plain `.mtlx` documents, the loader accepts `.mtlx.zip` archives that bundle a
 * document with its textures.
 *
 * ```js
 * const loader = new MaterialXLoader().setPath( SAMPLE_PATH );
 * const { materials } = await loader.loadAsync( 'standard_surface_brass_tiled.mtlx' );
 * ```
 */
class MaterialXLoader extends Loader {

	// `Loader` has no TS types (three.js ships none for these subpaths), so these
	// inherited fields aren't visible to the type checker — declare them explicitly.
	declare manager: any;
	declare path: string;

	private archiveDisposer: ( () => void ) | null;

	/**
	 * Constructs a new MaterialX loader.
	 *
	 * @param manager - The loading manager.
	 */
	constructor( manager?: any ) {

		super( manager );

		/**
		 * Releases the resources of the last loaded archive, if any.
		 */
		this.archiveDisposer = null;

	}

	/**
	 * Frees the resources of the last loaded archive.
	 */
	dispose(): this {

		if ( this.archiveDisposer ) {

			this.archiveDisposer();
			this.archiveDisposer = null;

		}

		return this;

	}

	/**
	 * Starts loading from the given URL and passes the loaded MaterialX asset
	 * to the `onLoad()` callback.
	 */
	load(
		url: string,
		onLoad: ( result: MaterialXParseResult ) => void,
		onProgress?: ( event: ProgressEvent ) => void,
		onError?: ( err: unknown ) => void,
		options: MaterialXLoaderParseOptions = {},
	): this {

		const _onError = function ( e: unknown ) {

			if ( onError ) {

				onError( e );

			} else {

				console.error( e );

			}

		};

		new FileLoader( this.manager )
			.setPath( this.path )
			.setResponseType( 'arraybuffer' )
			.load( url, ( data: any ) => {

				try {

					onLoad( this.parseBuffer( data, url, options ) );

				} catch ( e ) {

					_onError( e );

				}

			}, onProgress, _onError );

		return this;

	}

	/**
	 * Async version of {@link MaterialXLoader#load}. The progress callback can be
	 * omitted and the parse options passed as the second argument instead.
	 */
	loadAsync(
		url: string,
		onProgress?: ( ( event: ProgressEvent ) => void ) | MaterialXLoaderParseOptions,
		options: MaterialXLoaderParseOptions = {},
	): Promise<MaterialXParseResult> {

		if ( onProgress && typeof onProgress === 'object' ) {

			options = onProgress;
			onProgress = undefined;

		}

		return new Promise( ( resolve, reject ) => {

			this.load( url, resolve, onProgress as ( event: ProgressEvent ) => void, reject, options );

		} );

	}

	/**
	 * Parses a raw MaterialX document or a `.mtlx.zip` archive and returns the resulting materials.
	 */
	parseBuffer( data: ArrayBuffer | Uint8Array | string, url = '', options: MaterialXLoaderParseOptions = {} ): MaterialXParseResult {

		this.dispose();

		let text: string;
		let archiveResolver: ( ( uri: string ) => string | null ) | null = null;

		if ( data && ( isZipBuffer( data as ArrayBuffer | Uint8Array ) || /\.mtlx\.zip$/i.test( url ) ) ) {

			const archive = readMtlxArchive( data as ArrayBuffer | Uint8Array );
			text = archive.text;

			const resolver = createArchiveResolver( archive.files );
			archiveResolver = resolver.resolve;
			this.archiveDisposer = resolver.dispose;

		} else if ( typeof data === 'string' ) {

			text = data;

		} else if ( data instanceof Uint8Array ) {

			text = _textDecoder.decode( data );

		} else {

			text = _textDecoder.decode( new Uint8Array( data ) );

		}

		return this.parse( text, {
			...options,
			archiveResolver,
			path: options.path || getResourcePath( this.path, url ),
		} );

	}

	/**
	 * Parses the given MaterialX document and returns the resulting materials
	 * together with the translation log.
	 */
	parse( text: string, options: MaterialXLoaderParseOptions = {} ): MaterialXParseResult {

		const log = new MaterialXLog();

		const document = new MaterialXDocument( this.manager, options.path || this.path, log, options.archiveResolver || null, options.uvSpace );
		const result = document.parse( text, options.materialName || null, {
			interfaceValidator: options.interfaceValidator,
		} );

		if ( options.throwOnErrors !== false && log.errors.length > 0 ) {

			const details = log.errors.map( ( error ) => error.message ).join( ' ' );
			throw new Error( `THREE.MaterialXLoader: MaterialX translation failed with ${log.errors.length} error(s). ${details}` );

		}

		return result;

	}

}

export { MaterialXLoader };
