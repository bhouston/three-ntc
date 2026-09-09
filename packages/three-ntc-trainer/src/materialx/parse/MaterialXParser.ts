export interface MaterialXNodeLike {
	nodePath: string;
	add( child: any ): void;
}

function parseMaterialXNodeTree(
	nodeXML: Element,
	createNode: ( nodeXML: Element, nodePath: string ) => any,
	addNode: ( node: any ) => void,
	nodePath = '',
): any {

	const materialXNode = createNode( nodeXML, nodePath );
	if ( materialXNode.nodePath ) {

		addNode( materialXNode );

	}

	for ( const childNodeXML of Array.from( nodeXML.children ) ) {

		const childMXNode = parseMaterialXNodeTree( childNodeXML, createNode, addNode, materialXNode.nodePath );
		materialXNode.add( childMXNode );

	}

	return materialXNode;

}

function parseMaterialXText(
	text: string,
	createNode: ( nodeXML: Element, nodePath: string ) => any,
	addNode: ( node: any ) => void,
): any {

	const rootXML = new DOMParser().parseFromString( text, 'application/xml' ).documentElement;
	return parseMaterialXNodeTree( rootXML, createNode, addNode );

}

export { parseMaterialXNodeTree, parseMaterialXText };
