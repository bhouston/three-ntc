export interface MaterialXNodeLike {
  nodePath: string;
  add(child: this): void;
}

function parseMaterialXNodeTree<T extends MaterialXNodeLike>(
  nodeXML: Element,
  createNode: (nodeXML: Element, nodePath: string) => T,
  addNode: (node: T) => void,
  nodePath = '',
): T {
  const materialXNode = createNode(nodeXML, nodePath);
  if (materialXNode.nodePath) {
    addNode(materialXNode);
  }

  for (const childNodeXML of Array.from(nodeXML.children)) {
    const childMXNode = parseMaterialXNodeTree(childNodeXML, createNode, addNode, materialXNode.nodePath);
    materialXNode.add(childMXNode);
  }

  return materialXNode;
}

function parseMaterialXText<T extends MaterialXNodeLike>(
  text: string,
  createNode: (nodeXML: Element, nodePath: string) => T,
  addNode: (node: T) => void,
): T {
  const rootXML = new DOMParser().parseFromString(text, 'application/xml').documentElement;
  return parseMaterialXNodeTree(rootXML, createNode, addNode);
}

export { parseMaterialXNodeTree, parseMaterialXText };
