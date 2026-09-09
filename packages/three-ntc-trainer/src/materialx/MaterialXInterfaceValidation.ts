import { MtlXLibrary } from './MaterialXNodeLibrary.js';
import { MaterialXLogCodes, type MaterialXLog } from './MaterialXLog.js';
import registryData from './MaterialXNodeInterfaceRegistry.js';

const SKIP_ELEMENTS = new Set( [ 'materialx', 'input', 'output' ] );
const CONTAINER_ELEMENTS = new Set( [ 'nodegraph' ] );

const LIBRARY_FALLBACK_OUTPUT = 'out';

function formatInputElement( inputNodeX: any ): string {

	const attrs = [];
	for ( const name of [ 'name', 'type', 'nodename', 'nodegraph', 'interfacename', 'output', 'value' ] ) {

		const value = inputNodeX.getAttribute( name );
		if ( value !== null && value !== '' ) {

			attrs.push( `${name}="${value}"` );

		}

	}

	return `<input ${attrs.join( ' ' )}>`;

}

function formatNodeElement( nodeX: any ): string {

	const attrs = [];
	for ( const name of [ 'name', 'type', 'nodedef' ] ) {

		const value = nodeX.getAttribute( name );
		if ( value !== null && value !== '' ) {

			attrs.push( `${name}="${value}"` );

		}

	}

	return `<${nodeX.element} ${attrs.join( ' ' )}>`;

}

function scoreNodedefCandidate( candidateName: string, nodeX: any ): number {

	const candidate = registryData.resolved[ candidateName ];
	if ( ! candidate ) return - 1;

	let score = 0;
	const nodeType = nodeX.type;
	const outputType = candidate.outputs.out || Object.values( candidate.outputs )[ 0 ];
	if ( nodeType && outputType === nodeType ) {

		score += 4;

	}

	for ( const child of nodeX.children ) {

		if ( child.element !== 'input' ) continue;
		const expectedType = candidate.inputs[ child.name ];
		if ( ! expectedType || ! child.type ) continue;
		if ( expectedType === child.type ) {

			score += 3;

		} else if ( typesCompatible( expectedType, child.type ) ) {

			score += 1;

		} else {

			score -= 2;

		}

	}

	return score;

}

function resolveInterface( nodeX: any ): any {

	const nodedefName = nodeX.getAttribute( 'nodedef' );
	if ( nodedefName && registryData.resolved[ nodedefName ] ) {

		return { source: 'nodedef', name: nodedefName, ...registryData.resolved[ nodedefName ] };

	}

	const candidates = registryData.byNode[ nodeX.element ] || [];
	if ( candidates.length > 0 ) {

		let selected = candidates[ 0 ];
		let bestScore = - Infinity;

		for ( const candidateName of candidates ) {

			const score = scoreNodedefCandidate( candidateName, nodeX );
			if ( score > bestScore ) {

				bestScore = score;
				selected = candidateName;

			}

		}

		const resolved = registryData.resolved[ selected ];
		if ( resolved ) {

			return { source: 'nodedef', name: selected, ...resolved };

		}

	}

	const libraryEntry = ( MtlXLibrary as any )[ nodeX.element ];
	if ( libraryEntry ) {

		const inputs = Object.fromEntries( libraryEntry.params.map( ( param: string ) => [ param, null ] ) );
		const outputs = nodeX.type ? { [ LIBRARY_FALLBACK_OUTPUT ]: nodeX.type } : {};
		return { source: 'library', name: nodeX.element, node: nodeX.element, inputs, outputs };

	}

	return null;

}

function resolveReferencedNode( inputNodeX: any ): any {

	const materialX = inputNodeX.materialX;
	const referencePath = inputNodeX.referencePath;
	if ( ! referencePath ) return null;

	return materialX.getMaterialXNode( referencePath ) || null;

}

function resolveSourceOutputType( sourceNodeX: any, outputName: string | null ): { interface: any; outputType: string | null; outputName: string | null } {

	if ( sourceNodeX.element === 'output' ) {

		return {
			interface: null,
			outputType: sourceNodeX.type,
			outputName: sourceNodeX.name || 'out',
		};

	}

	const sourceInterface = resolveInterface( sourceNodeX );
	if ( ! sourceInterface ) return { interface: null, outputType: null, outputName };

	if ( outputName ) {

		if ( sourceInterface.outputs[ outputName ] ) {

			return {
				interface: sourceInterface,
				outputType: sourceInterface.outputs[ outputName ],
				outputName,
			};

		}

		return {
			interface: sourceInterface,
			outputType: null,
			outputName,
		};

	}

	if ( sourceNodeX.type && sourceNodeX.type !== 'multioutput' ) {

		return {
			interface: sourceInterface,
			outputType: sourceNodeX.type,
			outputName: 'out',
		};

	}

	if ( sourceInterface.outputs.out ) {

		return {
			interface: sourceInterface,
			outputType: sourceInterface.outputs.out,
			outputName: 'out',
		};

	}

	const declaredOutputs = Object.entries( sourceInterface.outputs );
	if ( declaredOutputs.length === 1 ) {

		const [ singleName, singleType ] = declaredOutputs[ 0 ];
		return {
			interface: sourceInterface,
			outputType: singleType as string,
			outputName: singleName,
		};

	}

	if ( sourceNodeX.type ) {

		return {
			interface: sourceInterface,
			outputType: sourceNodeX.type,
			outputName: null,
		};

	}

	return { interface: sourceInterface, outputType: null, outputName: null };

}

function typesCompatible( expectedType: string | null | undefined, actualType: string | null | undefined ): boolean {

	if ( ! expectedType || ! actualType ) return true;
	if ( expectedType === actualType ) return true;
	if ( ( expectedType === 'float' && actualType === 'integer' ) || ( expectedType === 'integer' && actualType === 'float' ) ) {

		return true;

	}

	return false;

}

function validatePortConnection( inputNodeX: any, parentNodeX: any, log: MaterialXLog ): void {

	const sourceNodeX = resolveReferencedNode( inputNodeX );
	if ( ! sourceNodeX ) return;

	const { outputType } = resolveSourceOutputType( sourceNodeX, inputNodeX.output );
	const inputSnippet = formatInputElement( inputNodeX );

	if ( inputNodeX.output && outputType === null ) {

		log.add(
			MaterialXLogCodes.INVALID_OUTPUT_CONNECTION,
			`No output found for port connection: ${inputSnippet}`,
			parentNodeX.name,
		);
		return;

	}

	const inputType = inputNodeX.type;
	if ( inputType && outputType && typesCompatible( inputType, outputType ) === false ) {

		log.add(
			MaterialXLogCodes.TYPE_MISMATCH,
			`Mismatched types in port connection: ${inputSnippet}`,
			parentNodeX.name,
		);

	}

}

function validateNodeInputs( nodeX: any, log: MaterialXLog ): void {

	if ( nodeX.element === 'materialx' ) {

		for ( const child of nodeX.children ) {

			validateNodeInputs( child, log );

		}

		return;

	}

	if ( SKIP_ELEMENTS.has( nodeX.element ) ) return;
	if ( CONTAINER_ELEMENTS.has( nodeX.element ) ) {

		for ( const child of nodeX.children ) {

			validateNodeInputs( child, log );

		}

		return;

	}

	const nodeInterface = resolveInterface( nodeX );
	if ( nodeInterface ) {

		for ( const child of nodeX.children ) {

			if ( child.element !== 'input' ) continue;

			const inputName = child.name;
			if ( ! inputName ) continue;

			if ( nodeInterface.inputs[ inputName ] === undefined ) {

				log.add(
					MaterialXLogCodes.UNKNOWN_INPUT,
					`Node interface error: Input '${inputName}' doesn't match declaration: ${formatNodeElement( nodeX )}`,
					nodeX.name,
				);

			} else if (
				child.type &&
				nodeInterface.inputs[ inputName ] &&
				child.hasReference === false &&
				typesCompatible( nodeInterface.inputs[ inputName ], child.type ) === false
			) {

				log.add(
					MaterialXLogCodes.TYPE_MISMATCH,
					`Input '${inputName}' type '${child.type}' doesn't match nodedef type '${nodeInterface.inputs[ inputName ]}' on ${formatNodeElement( nodeX )}`,
					nodeX.name,
				);

			}

			if ( child.hasReference ) {

				validatePortConnection( child, nodeX, log );

			}

		}

	} else if ( ( MtlXLibrary as any )[ nodeX.element ] === undefined && nodeX.element !== 'surfacematerial' ) {

		// Defer unsupported-node warnings to compile-time behavior.
		for ( const child of nodeX.children ) {

			if ( child.element === 'input' && child.hasReference ) {

				validatePortConnection( child, nodeX, log );

			}

		}

	}

	for ( const child of nodeX.children ) {

		if ( child.element !== 'input' && child.element !== 'output' ) {

			validateNodeInputs( child, log );

		}

	}

}

function createStrictInterfaceValidator() {

	return function validateMaterialXInterfaces( rootNode: any, log: MaterialXLog ) {

		validateNodeInputs( rootNode, log );

	};

}

export { createStrictInterfaceValidator };
