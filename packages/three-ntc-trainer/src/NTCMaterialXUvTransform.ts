
// The raw, uncompiled MaterialX document graph API this module walks
// (`getChildByName`/`hasReference`/`referencePath`/`isConst`/`getVector`,
// plus `element`/`nodePath`/`children`/`uvSpace`/`getMaterialXNode`) lives in
// the MaterialX loader port (`./materialx/`, ported separately) - typed
// loosely here rather than importing its concrete types, since this module
// only ever touches a handful of duck-typed members.
type MaterialXDocument = any;
type MaterialXNode = any;

import { Matrix3 } from 'three';

/**
 * Infers the combined UV transform a MaterialX-authored albedo graph applies
 * before its image lookup(s) - e.g. a `<place2d>` doing tiling/rotation, or a
 * `<rotate2d>` chained with a plain `<multiply>`/`<add>` for scale/offset -
 * and folds it into a single `THREE.any` suitable for `NTCNodeMaterial`'s
 * `uvTransform` / `NTCTextureSource.bakeColorNodeToTexture`'s inverse bake
 * (see NTCFormat.js's `decodeUvTransform`/`encodeUvTransform` for how this
 * round-trips through a `.ntc` file).
 *
 * This is deliberately a *simple heuristic*, not a general MaterialX
 * evaluator:
 *
 * - It operates on the raw, uncompiled MaterialX document graph (the
 *   `MaterialXNode` wrapper - `getChildByName`/`hasReference`/`referencePath`
 *   /`isConst`/`getVector`, the same API `MaterialXDocument.js`'s
 *   `resolveSurfaceShaderNode` and `MaterialXCompileRegistry.js`'s
 *   `compileGltfTextureNode` already use), not the compiled TSL node tree -
 *   once compiled, a `<rotate2d>` is indistinguishable from any other
 *   `mul`/`add`/`sin`/`cos` arithmetic, so this must run before/instead of
 *   compilation.
 * - Only a fixed, small vocabulary of node categories is recognized:
 *   `place2d`, `rotate2d`, and the generic `add`/`subtract`/`multiply`/
 *   `divide` arithmetic nodes (the common hand-authored "offset2d"/"scale2d"
 *   pattern - MaterialX's standard library has no node literally named
 *   `offset2d`/`transform2d`; `place2d` *is* MaterialX's actual combined
 *   pivot/scale/rotate/offset transform node - see `mx_place2d` in
 *   `src/nodes/materialx/MaterialXNodes.js`).
 * - Only **hardcoded** parameter values are handled - any transform node
 *   whose parameter is itself wired to another node (a computed expression)
 *   or a `<...>` animation input aborts detection entirely, falling back to
 *   identity, rather than guessing. See `isDynamicChild`.
 * - For `add`/`subtract`/`multiply`/`divide`, only the common authoring
 *   pattern - `in1` carries the UV chain, `in2` is a hardcoded vector2/float
 *   operand - is recognized; a node with the operands reversed, or with both
 *   sides fed by graphs, is not.
 * - Conflicting transforms across different channels (e.g. a differently
 *   transformed roughness map) are out of scope entirely - only the albedo
 *   chain is walked, and its transform is assumed to apply uniformly.
 * - The document's own `uvSpace` (see `MaterialXLoader`/`MaterialXDocument.
 *   js`) is honored: a document parsed with `uvSpace: 'top-left'` has an
 *   implicit Y-flip between the mesh's raw UV and MaterialX's own (bottom-
 *   left-origin) `texcoord` convention, which every recognized node upstream
 *   of the image actually operated on - see `inferUvTransformFromImageNode`'s
 *   handling of the `texcoord` root.
 * - Animation and per-pixel/procedural UV manipulation are out of scope -
 *   only a plain affine (rotate/scale/offset) composition is representable
 *   as a `any` in the first place.
 *
 * Falls back to an identity `any` (this module's stated default) the
 * moment any of the above is violated, rather than partially applying a
 * transform it isn't confident about.
 */
function inferAlbedoUvTransform( materialX: MaterialXDocument, surfaceShaderNode: MaterialXNode, options: { baseColorInputName?: string; maxNodes?: number } = {} ): any {

	const { baseColorInputName = 'base_color', maxNodes = 64 } = options;

	if ( ! surfaceShaderNode ) return new Matrix3();

	const baseColorChild = surfaceShaderNode.getChildByName( baseColorInputName );
	if ( ! baseColorChild ) return new Matrix3();

	const imageNode = findImageNode( materialX, baseColorChild, maxNodes );
	if ( ! imageNode ) return new Matrix3();

	return inferUvTransformFromImageNode( materialX, imageNode );

}

/**
 * Breadth-first search, starting at whatever node `rootChild` (an input
 * wrapper, e.g. the surface shader's `base_color` input) connects to, for
 * the nearest `<image>`/`<tiledimage>` node - the only node categories that
 * actually consume a `texcoord` the way this module cares about. Bounded by
 * `maxNodes` since a MaterialX graph can be arbitrarily large/cyclic-looking
 * through interface references; this only needs to answer "is there an
 * image lookup somewhere feeding albedo, and if so which one", not
 * enumerate the whole graph.
 */
function findImageNode( materialX: MaterialXDocument, rootChild: MaterialXNode, maxNodes: number ): MaterialXNode | null {

	const root = resolveUpstreamNode( materialX, rootChild );
	if ( ! root ) return null;

	const queue: MaterialXNode[] = [ root ];
	const visited = new Set<string>();
	let visitedCount = 0;

	while ( queue.length > 0 && visitedCount < maxNodes ) {

		const current = queue.shift();
		if ( ! current || visited.has( current.nodePath ) ) continue;

		visited.add( current.nodePath );
		visitedCount ++;

		if ( current.element === 'image' || current.element === 'tiledimage' ) return current;

		for ( const input of current.children ) {

			const upstream = resolveUpstreamNode( materialX, input );
			if ( upstream ) queue.push( upstream );

		}

	}

	return null;

}

/**
 * Walks backward from an `<image>`/`<tiledimage>` node's `texcoord` input
 * through the recognized UV-transform node chain (see this module's doc
 * comment), concatenating each node's affine effect via `any.multiply`
 * in the order encountered - which, since this walk visits nodes from the
 * image lookup *back toward* the UV source, is exactly the right order for
 * `any.multiply` (`this = this * m`, right-multiply) to build up
 * `finalMatrix = nodeClosestToImage * ... * nodeClosestToSource`, matching
 * how the actual node graph evaluates (source first, image-adjacent node
 * last).
 */
function inferUvTransformFromImageNode( materialX: MaterialXDocument, imageNode: MaterialXNode, options: { maxSteps?: number } = {} ): any {

	const { maxSteps = 8 } = options;

	const matrix = new Matrix3();
	let child = imageNode.getChildByName( 'texcoord' );
	let steps = 0;

	while ( child && steps < maxSteps ) {

		const upstream = resolveUpstreamNode( materialX, child );
		if ( ! upstream ) break; // a plain/default texcoord input - nothing more to fold in

		if ( upstream.element === 'texcoord' || upstream.element === 'geomcolor' ) {

			// MaterialX's own texcoord convention is bottom-left-origin; a
			// document parsed with `uvSpace: 'top-left'` (see MaterialXLoader.
			// js/MaterialXDocument.js's `getBottomLeftUvSpaceHelpers`) has a Y-
			// flip inserted right here, between the mesh's raw `uv()` and the
			// value every MaterialX node upstream of this point actually
			// operated on - so it has to be folded in as the innermost
			// (source-closest) step of this matrix, or every recognized node's
			// otherwise-correct math is being fed the wrong input. A uniform-
			// scale pure rotation can look deceptively close to correct even
			// with this missing (a checker-pattern mirror-plus-rotation is easy
			// to mistake for a plain rotation by eye); a non-uniform scale
			// (place2d's `scale`) exposes the mismatch immediately.
			if ( upstream.element === 'texcoord' && materialX.uvSpace === 'top-left' ) {

				matrix.multiply( affineFromSamples( ( x, y ) => [ x, 1 - y ] ) );

			}

			break; // reached the UV source itself

		}

		const transformFn = buildNodeTransformFn( upstream );
		if ( ! transformFn ) return new Matrix3(); // unrecognized or non-constant node - abort to identity, see doc comment

		matrix.multiply( affineFromSamples( transformFn ) );

		const nextInputName = UV_FLOW_INPUT[ upstream.element ] || 'in';
		child = upstream.getChildByName( nextInputName );
		steps ++;

	}

	return matrix;

}

// Which input of each recognized node category carries the "rest of the UV
// chain" (as opposed to a hardcoded parameter) - used to keep walking
// upstream past a recognized node. `add`/`subtract`/`multiply`/`divide`
// default to MaterialX's own `in1` (this module only recognizes the
// authoring pattern where `in1` is the UV chain and `in2` is the constant
// operand - see this module's doc comment).
const UV_FLOW_INPUT: Record<string, string> = {
	rotate2d: 'in',
	place2d: 'texcoord',
	add: 'in1',
	subtract: 'in1',
	multiply: 'in1',
	divide: 'in1'
};

/**
 * Resolves the `MaterialXNode` an input wrapper (`child`) connects to -
 * `null` if it's a plain hardcoded value or simply absent. Mirrors
 * `MaterialXDocument.js`'s own `resolveSurfaceShaderNode` convention
 * (`hasReference` -> `materialX.getMaterialXNode(referencePath)`).
 *
 * Transparently unwraps any chain of `<output>` elements found along the
 * way - e.g. a `base_color` input pointing at `nodegraph="foo" output="out"`
 * resolves first to that nodegraph's `<output name="out" nodename="...">`
 * element, which is itself a pure passthrough: its own connection lives in
 * a `nodename`/`nodegraph`+`output` attribute *on itself*, not a child
 * `<input>` (see `MaterialXNode.referencePath`'s `nodeName`-only branch) -
 * so without this unwrap, a graph whose albedo is exposed through a
 * nodegraph output (the common case - see the `standard_surface_rotate2d_
 * test.mtlx` sample) would dead-end on the `<output>` node itself instead of
 * reaching the `<image>`/transform nodes behind it.
 */
function resolveUpstreamNode( materialX: MaterialXDocument, child: MaterialXNode ): MaterialXNode | null {

	let node = ( child && child.hasReference ) ? materialX.getMaterialXNode( child.referencePath ) : null;

	let guard = 0;
	while ( node && node.element === 'output' && node.hasReference && guard < 8 ) {

		node = materialX.getMaterialXNode( node.referencePath );
		guard ++;

	}

	return node || null;

}

/**
 * True for an input that's present but *not* a plain hardcoded value - wired
 * to another node/nodegraph/interface, and therefore potentially computed or
 * animated. An absent input (using its node-type default) is not dynamic -
 * there's nothing to be uncertain about.
 */
function isDynamicChild( child: MaterialXNode | null | undefined ): boolean {

	return Boolean( child ) && ! child.isConst && child.hasReference;

}

function constVector( child: MaterialXNode | null | undefined, fallback: number[] ): number[] {

	if ( ! child || ! child.isConst ) return fallback;

	const vector = child.getVector();
	return vector.length > 0 ? vector : fallback;

}

function constFloat( child: MaterialXNode | null | undefined, fallback: number ): number {

	return constVector( child, [ fallback ] )[ 0 ];

}

function constVec2( child: MaterialXNode | null | undefined, fallback: number[] ): [ number, number ] {

	const vector = constVector( child, fallback );
	return vector.length >= 2 ? [ vector[ 0 ], vector[ 1 ] ] : [ vector[ 0 ], vector[ 0 ] ];

}

/**
 * Plain scalar reimplementation of `mx_rotate2d` (see
 * `src/nodes/materialx/MaterialXCore.js`) - degrees, not radians, matching
 * MaterialX's own `<rotate2d>` node.
 */
function rotate2D( x: number, y: number, amountDegrees: number ): [ number, number ] {

	const radians = amountDegrees * Math.PI / 180;
	const ca = Math.cos( radians );
	const sa = Math.sin( radians );

	return [ ca * x + sa * y, ca * y - sa * x ];

}

/**
 * Plain scalar reimplementation of `mx_place2d` (see
 * `src/nodes/materialx/MaterialXNodes.js`) - only the two discrete
 * `operationorder` cases (`0` = srt, nonzero = trs) are handled, matching
 * `mx_place2d`'s own `typeof operationorder === 'number'` branch; this
 * module only ever deals in hardcoded constants, so `operationorder` is
 * always a plain number here.
 */
function place2D( x: number, y: number, pivot: [ number, number ], scale: [ number, number ], rotateDegrees: number, offset: [ number, number ], operationorder: number ): [ number, number ] {

	const [ px, py ] = pivot;
	const [ sx, sy ] = scale;
	const [ ox, oy ] = offset;
	const cx = x - px;
	const cy = y - py;

	if ( Math.abs( operationorder ) <= Number.EPSILON ) {

		// srt: rotate2d(centered / scale, rotate) - offset + pivot
		const [ rx, ry ] = rotate2D( cx / sx, cy / sy, rotateDegrees );
		return [ rx - ox + px, ry - oy + py ];

	}

	// trs: rotate2d(centered - offset, rotate) / scale + pivot
	const [ rx, ry ] = rotate2D( cx - ox, cy - oy, rotateDegrees );
	return [ rx / sx + px, ry / sy + py ];

}

const ARITHMETIC_OPS: Record<string, ( v: number, o: number ) => number> = {
	add: ( v, o ) => v + o,
	subtract: ( v, o ) => v - o,
	multiply: ( v, o ) => v * o,
	divide: ( v, o ) => v / o
};

type Affine2DFn = ( x: number, y: number ) => [ number, number ];

/**
 * Builds a plain `(x, y) => [x2, y2]` function for one recognized,
 * all-constant UV-transform node, or `null` if `node` isn't a recognized
 * category or has a non-constant parameter (see `isDynamicChild`) - the
 * caller (`inferUvTransformFromImageNode`) treats `null` as "abort to
 * identity".
 */
function buildNodeTransformFn( node: MaterialXNode ): Affine2DFn | null {

	switch ( node.element ) {

		case 'rotate2d': {

			const amountChild = node.getChildByName( 'amount' );
			if ( isDynamicChild( amountChild ) ) return null;

			const amount = constFloat( amountChild, 0 );
			return ( x, y ) => rotate2D( x, y, amount );

		}

		case 'place2d': {

			const pivotChild = node.getChildByName( 'pivot' );
			const scaleChild = node.getChildByName( 'scale' );
			const rotateChild = node.getChildByName( 'rotate' );
			const offsetChild = node.getChildByName( 'offset' );
			const operationOrderChild = node.getChildByName( 'operationorder' );

			if ( [ pivotChild, scaleChild, rotateChild, offsetChild, operationOrderChild ].some( isDynamicChild ) ) return null;

			const pivot = constVec2( pivotChild, [ 0, 0 ] );
			const scale = constVec2( scaleChild, [ 1, 1 ] );
			const rotate = constFloat( rotateChild, 0 );
			const offset = constVec2( offsetChild, [ 0, 0 ] );
			const operationorder = constFloat( operationOrderChild, 0 );

			return ( x, y ) => place2D( x, y, pivot, scale, rotate, offset, operationorder );

		}

		case 'add':
		case 'subtract':
		case 'multiply':
		case 'divide': {

			const in2Child = node.getChildByName( 'in2' );
			if ( isDynamicChild( in2Child ) ) return null;

			const identity = ( node.element === 'multiply' || node.element === 'divide' ) ? [ 1, 1 ] : [ 0, 0 ];
			const operand = constVec2( in2Child, identity );
			const op = ARITHMETIC_OPS[ node.element ];

			return ( x, y ) => [ op( x, operand[ 0 ] ), op( y, operand[ 1 ] ) ];

		}

		default:
			return null;

	}

}

/**
 * Recovers the 2D affine `any` for a plain `(x, y) => [x2, y2]`
 * function, by evaluating it at three points - `(0,0)`, `(1,0)`, `(0,1)` -
 * rather than hand-deriving the matrix algebraically for every node type:
 * `fn(0,0)` is the translation, and `fn(1,0) - fn(0,0)` / `fn(0,1) -
 * fn(0,0)` are the linear part's two columns. Valid for any `fn` that's
 * genuinely affine in `(x, y)` - true of every node `buildNodeTransformFn`
 * recognizes (rotate/scale/offset compositions), since none of them involve
 * `x`/`y` in a nonlinear way.
 */
function affineFromSamples( fn: Affine2DFn ): any {

	const [ ox, oy ] = fn( 0, 0 );
	const [ x1, y1 ] = fn( 1, 0 );
	const [ x2, y2 ] = fn( 0, 1 );

	return new Matrix3().set(
		x1 - ox, x2 - ox, ox,
		y1 - oy, y2 - oy, oy,
		0, 0, 1
	);

}

export { inferAlbedoUvTransform, inferUvTransformFromImageNode, findImageNode };
