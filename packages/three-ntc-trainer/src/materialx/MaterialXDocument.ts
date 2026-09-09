import {
	Texture,
	RepeatWrapping,
	ClampToEdgeWrapping,
	MirroredRepeatWrapping,
	ImageLoader,
	ImageBitmapLoader,
	Matrix3,
	Matrix4,
	MeshBasicNodeMaterial,
	MeshPhysicalNodeMaterial,
} from 'three/webgpu';

import {
	float,
	int,
	bool,
	sub,
	vec2,
	vec3,
	vec4,
	color,
	uv,
	mat3,
	mat4,
	element,
	mx_transform_uv,
	mx_srgb_texture_to_lin_rec709,
} from 'three/tsl';

import { MaterialXLogCodes, type MaterialXLog } from './MaterialXLog.js';
import { createMaterialXCompileRegistry, compileNodeFromRegistry, type MaterialXCompileContext } from './compile/MaterialXCompileRegistry.js';
import { parseMaterialXNodeTree, parseMaterialXText } from './parse/MaterialXParser.js';
import { getSurfaceMapper } from './MaterialXSurfaceMappings.js';
import { MtlXLibrary } from './MaterialXNodeLibrary.js';
import { mxHextileCoord, mxHextileComputeBlendWeights } from './MaterialXHextile.js';
import { toBooleanNode } from './MaterialXUtils.js';

const colorSpaceLib: Record<string, ( node: any ) => any> = {
	mx_srgb_texture_to_lin_rec709,
};

const DEFAULT_DOCUMENT_COLOR_SPACE = 'lin_rec709';
const IDENTITY_MAT3_VALUES = [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ];
const IDENTITY_MAT4_VALUES = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ];
const MATRIX_INVERSE_EPSILON = 1e-8;
const COMPILE_REGISTRY = createMaterialXCompileRegistry();
const TEXTURE_ADDRESS_MODE_WRAPPING: Record<string, number> = {
	constant: ClampToEdgeWrapping,
	clamp: ClampToEdgeWrapping,
	periodic: RepeatWrapping,
	mirror: MirroredRepeatWrapping,
};
const NODE_CLASS_BY_TYPE: Record<string, any> = {
	integer: int,
	float,
	vector2: vec2,
	vector3: vec3,
	vector4: vec4,
	color4: vec4,
	color3: color,
	boolean: null,
	matrix33: mat3,
	matrix44: mat4,
};
const OUTPUT_CHANNELS: Record<string, number> = {
	outx: 0,
	outr: 0,
	outy: 1,
	outg: 1,
	outz: 2,
	outb: 2,
	outw: 3,
	outa: 3,
};

function mxFlipUvY( uvNode: any ) {

	return vec2( element( uvNode, 0 ), sub( 1, element( uvNode, 1 ) ) );

}

function mxIdentityUv( uvNode: any ) {

	return uvNode;

}

function getBottomLeftUvSpaceHelpers( uvSpace: string ) {

	const helper = uvSpace === 'top-left' ? mxFlipUvY : mxIdentityUv;
	return {
		mxToBottomLeftUvSpace: helper,
		mxFromBottomLeftUvSpace: helper,
	};

}

function normalizeUvSpace( uvSpace: string | undefined ): 'bottom-left' | 'top-left' {

	if ( uvSpace === undefined || uvSpace === null ) return 'bottom-left';
	if ( uvSpace === 'bottom-left' || uvSpace === 'top-left' ) return uvSpace;
	throw new Error( `Unsupported MaterialX uvSpace "${uvSpace}". Expected "bottom-left" or "top-left".` );

}

function isSvgUri( uri: unknown ): boolean {

	if ( typeof uri !== 'string' ) return false;
	return /\.svg(?:$|[?#])/i.test( uri );

}

function normalizeTextureAddressMode( value: any ): string | null {

	if ( value === null || value === undefined || value === '' ) return 'periodic';

	const mode = value.trim().toLowerCase();
	return mode in TEXTURE_ADDRESS_MODE_WRAPPING ? mode : null;

}

function invertConstantMatrixValues( values: number[], size: number ): number[] | null {

	if ( ! Array.isArray( values ) || values.length !== size * size ) return null;

	if ( size === 3 ) {

		const matrix = new Matrix3().setFromArray( values );
		if ( Math.abs( matrix.determinant() ) < MATRIX_INVERSE_EPSILON ) return null;
		matrix.invert();
		// Convert Three.js internal column-major storage back to row-major literal order.
		return matrix.transpose().elements as unknown as number[];

	}

	if ( size === 4 ) {

		const matrix = new Matrix4().setFromArray( values );
		if ( Math.abs( matrix.determinant() ) < MATRIX_INVERSE_EPSILON ) return null;
		matrix.invert();
		// Convert Three.js internal column-major storage back to row-major literal order.
		return matrix.transpose().elements as unknown as number[];

	}

	return null;

}

function getOutputChannel( outputName: string ): number {

	return OUTPUT_CHANNELS[ outputName ] || 0;

}

function isChannelOutput( outputName: any ): boolean {

	return outputName in OUTPUT_CHANNELS;

}

class MaterialXNode {

	materialX: MaterialXDocument;
	nodeXML: Element;
	nodePath: string;
	parent: MaterialXNode | null;
	node: any;
	children: MaterialXNode[];

	constructor( materialX: MaterialXDocument, nodeXML: Element, nodePath = '' ) {

		this.materialX = materialX;
		this.nodeXML = nodeXML;
		this.nodePath = nodePath ? nodePath + '/' + this.name : this.name;
		this.parent = null;
		this.node = null;
		this.children = [];

	}

	get element(): string {

		return this.nodeXML.nodeName;

	}
	get nodeGraph(): string | null {

		return this.getAttribute( 'nodegraph' );

	}
	get nodeName(): string | null {

		return this.getAttribute( 'nodename' );

	}
	get interfaceName(): string | null {

		return this.getAttribute( 'interfacename' );

	}
	get output(): string | null {

		return this.getAttribute( 'output' );

	}
	get name(): string {

		return this.getAttribute( 'name' ) as string;

	}
	get type(): string {

		return this.getAttribute( 'type' ) as string;

	}
	get value(): string | null {

		return this.getAttribute( 'value' );

	}

	getNodeGraph(): MaterialXNode | null {

		let nodeX: MaterialXNode | null = this;
		while ( nodeX !== null ) {

			if ( nodeX.element === 'nodegraph' ) break;
			nodeX = nodeX.parent;

		}

		return nodeX;

	}

	getRoot(): MaterialXNode {

		let nodeX: MaterialXNode = this;
		while ( nodeX.parent !== null ) {

			nodeX = nodeX.parent;

		}

		return nodeX;

	}

	get referencePath(): string | null {

		let referencePath = null;
		if ( this.nodeGraph !== null && this.output !== null ) {

			referencePath = this.nodeGraph + '/' + this.output;

		} else if ( this.nodeName !== null || this.interfaceName !== null ) {

			const graphNode = this.getNodeGraph();
			const scopedReference = this.nodeName || this.interfaceName;
			if ( graphNode && scopedReference ) {

				referencePath = graphNode.nodePath + '/' + scopedReference;

			} else if ( this.nodeName !== null ) {

				// Surface-level nodename links can legitimately target top-level siblings.
				referencePath = this.nodeName;

			}

		}

		return referencePath;

	}

	get hasReference(): boolean {

		return this.referencePath !== null;

	}
	get isConst(): boolean {

		return this.element === 'input' && this.value !== null && this.type !== 'filename';

	}

	getColorSpaceNode(): ( ( node: any ) => any ) | null {

		const csSource = this.getAttribute( 'colorspace' );
		const csTarget = this.getRoot().getAttribute( 'colorspace' );
		if ( ! csSource || ! csTarget ) return null;
		const nodeName = `mx_${csSource}_to_${csTarget}`;
		return colorSpaceLib[ nodeName ] || null;

	}

	getTextureAddressMode( inputName: string ): string {

		const rawMode = this.getInputValueByName( inputName );
		const mode = normalizeTextureAddressMode( rawMode );
		if ( mode ) return mode;

		this.materialX.log.add(
			MaterialXLogCodes.INVALID_VALUE,
			`Unsupported texture address mode "${rawMode}" on input "${inputName}". Expected constant, clamp, periodic, or mirror.`,
			this.name,
		);
		return 'periodic';

	}

	getTextureAddressModes(): { u: string; v: string } {

		return {
			u: this.getTextureAddressMode( 'uaddressmode' ),
			v: this.getTextureAddressMode( 'vaddressmode' ),
		};

	}

	getTexture(): any {

		const filePrefix = this.getRecursiveAttribute( 'fileprefix' ) || '';
		const sourceURI = filePrefix + this.value;
		const resolvedURI = this.materialX.resolveTextureURI( sourceURI );
		const svgTexture = isSvgUri( resolvedURI );
		const textureSourceNode: any = this.parent && typeof this.parent.getTextureAddressModes === 'function' ? this.parent : this;
		const addressModes = textureSourceNode.getTextureAddressModes();
		const textureCacheKey = `${resolvedURI}|${addressModes.u}|${addressModes.v}`;

		if ( this.materialX.textureCache.has( textureCacheKey ) ) {

			return this.materialX.textureCache.get( textureCacheKey )!;

		}

		let loader: any = svgTexture ? this.materialX.imageLoader : this.materialX.textureLoader;
		if ( resolvedURI && ! svgTexture ) {

			const handler = this.materialX.manager.getHandler( resolvedURI );
			if ( handler !== null ) loader = handler;

		}

		const textureNode = new Texture();
		textureNode.wrapS = TEXTURE_ADDRESS_MODE_WRAPPING[ addressModes.u ] as any;
		textureNode.wrapT = TEXTURE_ADDRESS_MODE_WRAPPING[ addressModes.v ] as any;
		textureNode.flipY = false;
		this.materialX.textureCache.set( textureCacheKey, textureNode );

		const nodeName = this.name;
		const materialX = this.materialX;

		materialX.textureLoadPromises.push( new Promise<void>( ( resolveLoad ) => {

			loader.load( resolvedURI, ( imageData: any ) => {

				textureNode.image = imageData;
				textureNode.needsUpdate = true;
				resolveLoad();

			}, undefined, () => {

				materialX.log.add(
					MaterialXLogCodes.TEXTURE_LOAD_FAILED,
					`Failed to load texture "${resolvedURI}".`,
					nodeName,
				);
				resolveLoad();

			} );

		} ) );

		return textureNode;

	}

	getClassFromType( type: string ): any {

		return NODE_CLASS_BY_TYPE[ type ] || null;

	}

	toBooleanNode( node: any ): any {

		return toBooleanNode( node );

	}

	getNode( out: string | null = null ): any {

		if ( this.node !== null && out === null ) return this.node;

		let node: any;


		if ( ( this.element === 'separate2' || this.element === 'separate3' || this.element === 'separate4' ) && out ) {

			const inNode = this.getNodeByName( 'in' );
			return element( inNode, getOutputChannel( out ) );

		}

		const type = this.type;
		const channelRequested = this.element !== 'input' && this.element !== 'gltf_colorimage' && isChannelOutput( out );

		if ( this.isConst ) {

			if ( type === 'boolean' ) {

				const normalized = this.getValue().trim().toLowerCase();
				node = bool( normalized === 'true' || normalized === '1' );

			} else if ( type === 'matrix33' ) {

				node = this.getMatrix( 3 ) || mat3( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

			} else if ( type === 'matrix44' ) {

				node = this.getMatrix( 4 ) || mat4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );

			} else if ( type === 'string' ) {

				node = this.getValue();

			} else {

				const nodeClass = this.getClassFromType( type );
				node = nodeClass ? nodeClass( ...this.getVector() ) : float( 0 );

			}

		} else if ( this.hasReference ) {

			if ( this.element === 'output' && this.output && out === null ) out = this.output;
			let requestedOutput = out;
			// For nodegraph references, this input's `output` attribute selects the graph output
			// itself and should not be forwarded as an output selector on the resolved node.
			if ( this.element === 'input' && this.nodeGraph !== null && this.output !== null ) {

				requestedOutput = null;

			}

			const referenceNode = this.materialX.getMaterialXNode( this.referencePath! );

			if ( referenceNode ) {

				node = referenceNode.getNode( requestedOutput );

			} else {

				this.materialX.log.add(
					MaterialXLogCodes.MISSING_REFERENCE,
					`Missing MaterialX reference "${this.referencePath}" from "${this.name}".`,
					this.name,
				);
				node = float( 0 );

			}

		} else if ( this.element === 'input' && this.name === 'texcoord' && this.type === 'vector2' ) {

			let index = 0;
			const defaultGeomProp = this.getAttribute( 'defaultgeomprop' );
			if ( defaultGeomProp && /^UV(\d+)$/.test( defaultGeomProp ) ) {

				index = parseInt( defaultGeomProp.match( /^UV(\d+)$/ )![ 1 ], 10 );

			}

			node = this.materialX.compileContext.getTexcoordNode( index );

		} else {

			node = compileNodeFromRegistry( this, out, this.materialX.compileContext );

		}

		if ( node === null || node === undefined ) {

			this.materialX.log.add(
				MaterialXLogCodes.UNSUPPORTED_NODE,
				`Unsupported MaterialX node category "${this.element}" on "${this.name}".`,
				this.name,
			);
			node = float( 0 );

		}

		if ( channelRequested ) {

			node = element( node, getOutputChannel( out! ) );

		}

		const resolvedType = channelRequested ? 'float' : type;
		if ( resolvedType === 'boolean' ) {

			node = this.toBooleanNode( node );

		} else if ( resolvedType === 'string' ) {

			// String-typed inputs (for example transform* fromspace/tospace) are
			// valid scalar parameters and should pass through without numeric casting.
			node = typeof node === 'string' ? node : this.getValue();

		} else {

			const nodeToTypeClass = this.getClassFromType( resolvedType );
			if ( nodeToTypeClass !== null ) {

				node = nodeToTypeClass( node );

			} else if ( resolvedType !== null && resolvedType !== undefined && resolvedType !== 'multioutput' ) {

				this.materialX.log.add(
					MaterialXLogCodes.INVALID_VALUE,
					`Unexpected type "${resolvedType}" on node "${this.name}".`,
					this.name,
				);
				node = float( 0 );

			}

		}

		if ( node && typeof node === 'object' ) {

			node.name = this.name;

		}

		this.node = node;
		return node;

	}

	getChildByName( name: string ): MaterialXNode | undefined {

		for ( const input of this.children ) {

			if ( input.name === name ) return input;

		}

		return undefined;

	}

	getNodes(): Record<string, any> {

		const nodes: Record<string, any> = {};
		for ( const input of this.children ) {

			const value = input.getNode( input.output );
			nodes[ input.name ] = value;

		}

		return nodes;

	}

	getNodeByName( name: string ): any {

		const child = this.getChildByName( name );
		return child ? child.getNode( child.output ) : undefined;

	}

	getInputValueByName( name: string ): string | null {

		const child = this.getChildByName( name );
		return child ? child.value : null;

	}

	getNodesByNames( ...names: string[] ): any[] {

		const nodes = [];
		for ( const name of names ) {

			const nodeValue = this.getNodeByName( name );
			nodes.push( nodeValue );

		}

		return nodes;

	}

	getValue(): string {

		return this.value ? this.value.trim() : '';

	}

	getVector(): number[] {

		const vector = [];
		for ( const val of this.getValue().split( /[,|\s]/ ) ) {

			if ( val !== '' ) vector.push( Number( val.trim() ) );

		}

		return vector;

	}

	getMatrix( size: number ): any {

		const vector = this.getVector();
		const expectedLength = size * size;
		if ( vector.length !== expectedLength ) return null;
		// MaterialX matrix values are serialized in column-major order.
		// Reorder to row-major before constructing TSL matrix nodes so
		// transformmatrix semantics match MaterialXJS and MaterialXView.
		const reordered = [];
		for ( let row = 0; row < size; row += 1 ) {

			for ( let column = 0; column < size; column += 1 ) {

				reordered.push( vector[ column * size + row ] );

			}

		}

		return size === 3 ? mat3( ...reordered ) : mat4( ...reordered );

	}

	getAttribute( name: string ): string | null {

		const value = this.nodeXML.getAttribute( name );
		if ( value === null && this.element === 'materialx' && name === 'colorspace' ) {

			return DEFAULT_DOCUMENT_COLOR_SPACE;

		}

		return value;

	}

	getRecursiveAttribute( name: string ): string | null {

		let attribute = this.nodeXML.getAttribute( name );
		if ( attribute === null && this.parent !== null ) {

			attribute = this.parent.getRecursiveAttribute( name );

		}

		return attribute;

	}

	setMaterial( material: any ): void {

		const mapper = getSurfaceMapper( this.element );
		if ( mapper ) {

			mapper.apply( material, this.getNodes(), this.materialX.log, this.name );

		} else {

			this.materialX.log.add(
				MaterialXLogCodes.UNSUPPORTED_NODE,
				`Unsupported MaterialX node category "${this.element}" on "${this.name}".`,
				this.name,
			);

		}

	}

	toBasicMaterial(): any {

		const material = new MeshBasicNodeMaterial();
		material.name = this.name;

		for ( const nodeX of this.children.toReversed() ) {

			if ( nodeX.name === 'out' ) {

				material.colorNode = nodeX.getNode();
				break;

			}

		}

		return material;

	}

	resolveSurfaceShaderNode( nodeX: MaterialXNode ): MaterialXNode | null {

		if ( nodeX.hasReference ) {

			return this.materialX.getMaterialXNode( nodeX.referencePath! ) || null;

		}

		if ( nodeX.nodeName ) {

			return this.materialX.getMaterialXNode( nodeX.nodeName ) || null;

		}

		return null;

	}

	toPhysicalMaterial(): any {

		const material: any = new MeshPhysicalNodeMaterial();
		material.name = this.name;

		for ( const nodeX of this.children ) {

			const shaderProperties = this.resolveSurfaceShaderNode( nodeX );
			if ( shaderProperties === null ) {

				this.materialX.log.add(
					MaterialXLogCodes.MISSING_REFERENCE,
					`Missing MaterialX reference "${nodeX.referencePath || nodeX.nodeName || '(unknown)'}" from "${nodeX.name}".`,
					nodeX.name,
				);
				continue;

			}

			shaderProperties.setMaterial( material );

			// Keep the raw MaterialX graph reachable for tools that need to
			// inspect it before the compiled TSL graph loses authoring intent.
			material.materialXSurfaceShaderNode = shaderProperties;
			material.materialXDocument = this.materialX;

		}

		return material;

	}

	toMaterials( materialName: string | null = null ): Record<string, any> {

		const materials: Record<string, any> = {};
		const surfaceMaterials = this.children.filter( ( nodeX ) => nodeX.element === 'surfacematerial' );

		let selectedSurfaceMaterials = surfaceMaterials;
		if ( materialName ) {

			selectedSurfaceMaterials = surfaceMaterials.filter( ( nodeX ) => nodeX.name === materialName );

			if ( selectedSurfaceMaterials.length === 0 ) {

				this.materialX.log.add(
					MaterialXLogCodes.MISSING_MATERIAL,
					`Could not find surfacematerial named "${materialName}".`,
				);

			}

		}

		for ( const nodeX of selectedSurfaceMaterials ) {

			const material = nodeX.toPhysicalMaterial();
			materials[ material.name ] = material;

		}

		if ( Object.keys( materials ).length === 0 ) {

			for ( const nodeX of this.children ) {

				if ( nodeX.element === 'nodegraph' ) {

					const material = nodeX.toBasicMaterial();
					materials[ material.name ] = material;

				}

			}

		}

		return materials;

	}

	add( materialXNode: MaterialXNode ): void {

		materialXNode.parent = this;
		this.children.push( materialXNode );

	}

}

export interface MaterialXParseResult {
	materials: Record<string, any>;
	log: import('./MaterialXLog.js').MaterialXLogEntry[];
	errors: import('./MaterialXLog.js').MaterialXLogEntry[];
	warnings: import('./MaterialXLog.js').MaterialXLogEntry[];
	texturesReady: Promise<void>;
}

class MaterialXDocument {

	manager: any;
	path: string;
	log: MaterialXLog;
	archiveResolver: ( ( uri: string ) => string | null ) | null;
	uvSpace: 'bottom-left' | 'top-left';
	nodesXLib: Map<string, MaterialXNode>;
	imageLoader: any;
	textureLoader: any;
	textureCache: Map<string, any>;
	textureLoadPromises: Promise<void>[];
	compileContext: MaterialXCompileContext;

	constructor( manager: any, path: string, log: MaterialXLog, archiveResolver: ( ( uri: string ) => string | null ) | null = null, uvSpace?: string ) {

		this.manager = manager;
		this.path = path;
		this.log = log;
		this.archiveResolver = archiveResolver;
		this.uvSpace = normalizeUvSpace( uvSpace );

		this.nodesXLib = new Map();
		this.imageLoader = new ImageLoader( manager );
		this.imageLoader.setPath( path );
		this.textureLoader = new ImageBitmapLoader( manager );
		this.textureLoader.setOptions( { imageOrientation: 'none' } );
		this.textureLoader.setPath( path );
		this.textureCache = new Map();
		this.textureLoadPromises = [];
		const bottomLeftUvSpaceHelpers = getBottomLeftUvSpaceHelpers( this.uvSpace );

		this.compileContext = {
			compileRegistry: COMPILE_REGISTRY,
			nodeLibrary: MtlXLibrary,
			...bottomLeftUvSpaceHelpers,
			getTexcoordNode: ( index = 0 ) => bottomLeftUvSpaceHelpers.mxToBottomLeftUvSpace( uv( index ) ),
			mxTransformUv: mx_transform_uv,
			mxHextileCoord,
			mxHextileComputeBlendWeights,
			invertConstantMatrixValues,
			IDENTITY_MAT3_VALUES,
			IDENTITY_MAT4_VALUES,
		};

	}

	resolveTextureURI( uri: string ): string {

		if ( this.archiveResolver ) {

			const archiveURI = this.archiveResolver( uri );
			if ( archiveURI ) return archiveURI;

		}

		return uri;

	}

	addMaterialXNode( materialXNode: MaterialXNode ): void {

		this.nodesXLib.set( materialXNode.nodePath, materialXNode );

	}

	getMaterialXNode( ...names: string[] ): MaterialXNode | undefined {

		return this.nodesXLib.get( names.join( '/' ) );

	}

	parseNode( nodeXML: Element, nodePath = '' ): MaterialXNode {

		return parseMaterialXNodeTree(
			nodeXML,
			( childNodeXML: Element, childNodePath: string ) => new MaterialXNode( this, childNodeXML, childNodePath ),
			( materialXNode: MaterialXNode ) => this.addMaterialXNode( materialXNode ),
			nodePath,
		);

	}

	parse( text: string, materialName: string | null = null, options: { interfaceValidator?: ( rootNode: MaterialXNode, log: MaterialXLog ) => void } = {} ): MaterialXParseResult {

		const rootNode: MaterialXNode = parseMaterialXText(
			text,
			( childNodeXML: Element, childNodePath: string ) => new MaterialXNode( this, childNodeXML, childNodePath ),
			( materialXNode: MaterialXNode ) => this.addMaterialXNode( materialXNode ),
		);

		if ( options.interfaceValidator ) {

			options.interfaceValidator( rootNode, this.log );

		}

		const materials = rootNode.toMaterials( materialName );
		return {
			materials,
			log: this.log.entries,
			errors: this.log.errors,
			warnings: this.log.warnings,
			texturesReady: Promise.all( this.textureLoadPromises ).then( () => undefined ),
		};

	}

}

export { MaterialXDocument, MaterialXNode };
