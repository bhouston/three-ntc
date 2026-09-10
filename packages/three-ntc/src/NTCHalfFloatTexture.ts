import { DataArrayTexture, DataTexture, DataUtils, HalfFloatType, LinearFilter, LinearMipmapLinearFilter, NearestFilter, NearestMipmapLinearFilter, RepeatWrapping, RGBAFormat } from 'three';
import { selectFeatureLevel } from './NTCMipBands.js';

/** One decoded stored feature-grid level (see NTCLoader.ts's `decodeLevel`). */
export interface NTCGrid {
	width: number;
	height: number;
	channels: number;
	data: Float32Array;
}

/** The subset of a decoded NTC CPU model `buildMipChainTexture` needs. */
export interface NTCMipChainModel {
	grids: NTCGrid[];
	lowResGrids?: NTCGrid[];
	mipsPerLevel: number;
}

/**
 * Packs a flat array of `channels`-per-texel float values (`channels` may be
 * less than 4, e.g. a single-channel roughness grid) into an RGBA half-float
 * `Uint16Array`, zero-padding any channel beyond `channels` - shared by
 * every neural-* area that authors a filterable/tileable latent-grid or
 * teacher-atlas texture (neural-texture's `buildMipChainTexture` below, the
 * NeuralAppearanceLoader's per-level textures).
 */
function packHalfFloatRGBA( data: Float32Array | number[], channels = 4 ): Uint16Array {

	const texelCount = data.length / channels;
	const packed = new Uint16Array( texelCount * 4 );

	for ( let p = 0; p < texelCount; p ++ ) {

		for ( let c = 0; c < 4; c ++ ) {

			const value = c < channels ? data[ p * channels + c ] : 0;
			packed[ p * 4 + c ] = DataUtils.toHalfFloat( value );

		}

	}

	return packed;

}

/**
 * Builds an RGBA16F `DataTexture` from a flat float array, pre-configured
 * for bilinear-filtered, tileable latent-grid sampling: half-float (not
 * full float, since RGBA32F isn't filterable under WebGPU without an opt-in
 * feature), repeat-or-clamp wrap, linear filtering, no mipmaps.
 */
function createHalfFloatLatentTexture( data: Float32Array | number[], width: number, height: number, { channels = 4, wrap = RepeatWrapping, filter = LinearFilter }: { channels?: number; wrap?: any; filter?: any } = {} ): any {

	const packed = packHalfFloatRGBA( data, channels );
	const texture = new DataTexture( packed, width, height, RGBAFormat, HalfFloatType );

	texture.wrapS = wrap;
	texture.wrapT = wrap;
	texture.magFilter = filter;
	texture.minFilter = filter;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;

	return texture;

}

/**
 * Builds one raw, unfiltered (`NearestFilter`) `DataTexture` per stored
 * grid level - used only by the `positionalEncoding` decoder path
 * (NTCDecoderTSL.js), which needs exact individual texel values (its own
 * "4-tap learned interpolation", NVIDIA paper Section 4.3.1) rather than a
 * hardware-bilinear/trilinear blend. `NearestFilter` + `RepeatWrapping`
 * means sampling at a texel center's UV returns that exact stored value,
 * with wraparound handled by the sampler - matching NTCGPUComputeTSL.js's
 * training kernel's own `wrapIndexTSL` addressing without needing to
 * reproduce that index math here.
 */
function updateLevelTextures(cpuModel: NTCMipChainModel, textures: any[]): void {
	const grids = [...cpuModel.grids, ...(cpuModel.lowResGrids || [])];
	if (textures.length !== grids.length) throw new Error('NTC grid count changed; rebuild the material.');
	for(let g=0;g<grids.length;g++) {
		const grid=grids[g], texture=textures[g];
		const depth=Math.ceil(grid.channels/4);
		if(texture.image.width!==grid.width || texture.image.height!==grid.height || texture.image.depth!==depth)
			throw new Error('NTC grid shape changed; rebuild the material.');
		const packed=texture.image.data;
		for(let p=0;p<grid.width*grid.height;p++) for(let c=0;c<grid.channels;c++) {
			packed[(Math.floor(c/4)*grid.width*grid.height+p)*4+c%4]=DataUtils.toHalfFloat(grid.data[p*grid.channels+c]);
		}
		texture.needsUpdate=true;
	}
}

function buildLevelTextures(cpuModel: NTCMipChainModel): any[] {
	const textures=[...cpuModel.grids,...(cpuModel.lowResGrids || [])].map(grid=>{
		const depth=Math.ceil(grid.channels/4);
		const texture=new DataArrayTexture(new Uint16Array(grid.width*grid.height*depth*4),grid.width,grid.height,depth);
		texture.type=HalfFloatType;texture.format=RGBAFormat;
		texture.wrapS=RepeatWrapping;texture.wrapT=RepeatWrapping;
		texture.magFilter=NearestFilter;texture.minFilter=NearestFilter;
		texture.generateMipmaps=false;
		return texture;
	});
	updateLevelTextures(cpuModel,textures);
	return textures;
}

export { updateLevelTextures };

interface RawMip {
	data: Float32Array;
	width: number;
	height: number;
}

/**
 * Wrap-aware 2x2 box-filter downsample of one `channels`-per-texel Float32
 * grid to `floor(width/2) x floor(height/2)` - the exact size a real GPU mip
 * chain requires of the next level down. Wraps at the edges (rather than
 * clamping) so the result stays seamlessly tileable, matching the latent
 * grids' own `RepeatWrapping`. Degenerates to a same-size copy once a
 * dimension has already reached 1 (a 1x1 input can't be halved further -
 * `buildStoredLevelMipPyramid` relies on this to pad out a band that runs
 * past where the source data bottoms out).
 */
function boxFilterDownsampleHalving( data: Float32Array, width: number, height: number, channels: number ): RawMip {

	if ( width <= 1 && height <= 1 ) return { data, width, height };

	const outWidth = Math.max( 1, Math.floor( width / 2 ) );
	const outHeight = Math.max( 1, Math.floor( height / 2 ) );
	const out = new Float32Array( outWidth * outHeight * channels );

	for ( let y = 0; y < outHeight; y ++ ) {

		const y0 = ( y * 2 ) % height;
		const y1 = ( y * 2 + 1 ) % height;

		for ( let x = 0; x < outWidth; x ++ ) {

			const x0 = ( x * 2 ) % width;
			const x1 = ( x * 2 + 1 ) % width;

			const p00 = ( y0 * width + x0 ) * channels;
			const p10 = ( y0 * width + x1 ) * channels;
			const p01 = ( y1 * width + x0 ) * channels;
			const p11 = ( y1 * width + x1 ) * channels;
			const o = ( y * outWidth + x ) * channels;

			for ( let c = 0; c < channels; c ++ ) {

				out[ o + c ] = ( data[ p00 + c ] + data[ p10 + c ] + data[ p01 + c ] + data[ p11 + c ] ) * 0.25;

			}

		}

	}

	return { data: out, width: outWidth, height: outHeight };

}

/**
 * Builds `stepCount` progressively-halved mips from one stored feature
 * level's own data (mip 0 of the pyramid is the level's data untouched) -
 * one stored level's band worth of physical GPU mips, see
 * `buildMipChainTexture`'s doc comment for why this (rather than a literal
 * duplicate) is how a stored level's single band gets spread across more
 * than one real mip level.
 */
function buildStoredLevelMipPyramid( grid: NTCGrid, stepCount: number ): RawMip[] {

	const mips: RawMip[] = [ { data: grid.data, width: grid.width, height: grid.height } ];

	for ( let i = 1; i < stepCount; i ++ ) {

		const prev = mips[ i - 1 ];
		mips.push( boxFilterDownsampleHalving( prev.data, prev.width, prev.height, grid.channels ) );

	}

	return mips;

}

/** One physical GPU mip level's unpacked (full Float32 precision) data. */
export interface MipChainLevel {
	data: Float32Array;
	width: number;
	height: number;
	channels: number;
}

/**
 * Builds the (unpacked, full Float32 precision) per-physical-mip data
 * `buildMipChainTexture` assembles into an RGBA half-float `DataTexture` -
 * factored out so tests can construct an exact CPU reference for a given
 * fractional LOD (the two bracketing integer physical mips' raw data, before
 * half-float rounding) without duplicating this box-filter/band-mapping
 * logic. See `buildMipChainTexture`'s doc comment for what each entry means;
 * this returns one `{ data, width, height, channels }` per physical mip
 * index (index 0 = finest), `channels` taken from the owning stored grid.
 */
function buildMipChainLevels( cpuModel: NTCMipChainModel ): MipChainLevel[] {

	const { grids, mipsPerLevel } = cpuModel;
	const baseWidth = grids[ 0 ].width;

	// Counted by repeated floor-halving (matching `computeGridLevels`'s own
	// `floor(resolution / 2)` step) rather than `1 + floor(log2(baseWidth))`,
	// which can undercount by one for an exact power of two due to
	// floating-point log2 landing a hair below the true integer.
	let totalMips = 1;
	for ( let w = baseWidth; w > 1; w = Math.floor( w / 2 ) ) totalMips ++;

	const levels: MipChainLevel[] = [];
	let cachedLevelIndex = - 1;
	let cachedPyramid: RawMip[] | null = null;

	for ( let p = 0; p < totalMips; p ++ ) {

		const levelIndex = selectFeatureLevel( p, grids.length, mipsPerLevel );

		if ( levelIndex !== cachedLevelIndex ) {

			cachedLevelIndex = levelIndex;
			const isLastLevel = levelIndex === grids.length - 1;
			const stepCount = isLastLevel ? totalMips - levelIndex * mipsPerLevel : mipsPerLevel;
			cachedPyramid = buildStoredLevelMipPyramid( grids[ levelIndex ], stepCount );

		}

		const bandStep = p - levelIndex * mipsPerLevel;
		const mip = cachedPyramid![ bandStep ];
		levels.push( { data: mip.data, width: mip.width, height: mip.height, channels: grids[ levelIndex ].channels } );

	}

	return levels;

}

/** Legacy storage container. Only band-start mips contain trained features.
 * Intermediate mips satisfy GPU allocation rules; decoding must not sample them.
 * New runtime paths use native per-level textures instead.
 */
function buildMipChainTexture( cpuModel: NTCMipChainModel, { interpolation = true }: { interpolation?: boolean } = {} ): any {

	const levels = buildMipChainLevels( cpuModel );
	const mipmaps = levels.map( ( level ) => ( {
		data: packHalfFloatRGBA( level.data, level.channels ),
		width: level.width,
		height: level.height
	} ) );

	const texture = new DataTexture( mipmaps[ 0 ].data, mipmaps[ 0 ].width, mipmaps[ 0 ].height, RGBAFormat, HalfFloatType );

	texture.mipmaps = mipmaps;
	texture.wrapS = RepeatWrapping;
	texture.wrapT = RepeatWrapping;
	texture.magFilter = interpolation ? LinearFilter : NearestFilter;
	texture.minFilter = interpolation ? LinearMipmapLinearFilter : NearestMipmapLinearFilter;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;

	return texture;

}

export { packHalfFloatRGBA, createHalfFloatLatentTexture, buildLevelTextures, buildMipChainLevels, buildMipChainTexture };
