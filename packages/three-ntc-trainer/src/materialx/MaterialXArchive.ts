import { unzipSync } from 'fflate';

const _textDecoder = new TextDecoder();

export interface MaterialXArchiveResult {
	text: string;
	mtlxPath: string;
	files: Map<string, Uint8Array>;
}

function normalizePath( path: string ): string {

	return path
		.split( '\\' ).join( '/' )
		.replace( /^\.?\//, '' )
		.replace( /^\/+/, '' );

}

function isZipBuffer( buffer: ArrayBuffer | Uint8Array ): boolean {

	const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array( buffer );
	return bytes.length >= 4 && bytes[ 0 ] === 0x50 && bytes[ 1 ] === 0x4b;

}

function readMtlxArchive( buffer: ArrayBuffer | Uint8Array ): MaterialXArchiveResult {

	const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array( buffer );
	const archive = unzipSync( bytes );

	const fileMap = new Map<string, Uint8Array>();
	let mtlxPath: string | null = null;

	for ( const path in archive ) {

		const normalizedPath = normalizePath( path );
		fileMap.set( normalizedPath, archive[ path ] );

		if ( normalizedPath.toLowerCase().endsWith( '.mtlx' ) ) {

			if ( mtlxPath !== null ) {

				throw new Error( 'THREE.MaterialXLoader: Invalid .mtlx.zip package. Exactly one .mtlx file is required.' );

			}

			mtlxPath = normalizedPath;

		}

	}

	if ( mtlxPath === null ) {

		throw new Error( 'THREE.MaterialXLoader: Invalid .mtlx.zip package. Missing .mtlx file.' );

	}

	const text = _textDecoder.decode( fileMap.get( mtlxPath ) );
	return { text, mtlxPath, files: fileMap };

}

function createArchiveResolver( files: Map<string, Uint8Array> ): { resolve: ( uri: string ) => string | null; dispose: () => void } {

	const objectUrlCache = new Map<string, string>();

	const getFile = ( uri: string ): Uint8Array | null => {

		const normalized = normalizePath( decodeURI( uri ) );
		if ( files.has( normalized ) ) return files.get( normalized )!;

		for ( const [ path, bytes ] of files ) {

			if ( path.endsWith( normalized ) ) return bytes;

		}

		return null;

	};

	const resolve = ( uri: string ): string | null => {

		if ( objectUrlCache.has( uri ) ) return objectUrlCache.get( uri )!;

		const bytes = getFile( uri );
		if ( ! bytes ) return null;

		const blob = new Blob( [ bytes as BlobPart ], { type: 'application/octet-stream' } );
		const objectUrl = URL.createObjectURL( blob );
		objectUrlCache.set( uri, objectUrl );
		return objectUrl;

	};

	const dispose = () => {

		for ( const objectUrl of objectUrlCache.values() ) {

			URL.revokeObjectURL( objectUrl );

		}

		objectUrlCache.clear();

	};

	return { resolve, dispose };

}

export { isZipBuffer, readMtlxArchive, createArchiveResolver };
