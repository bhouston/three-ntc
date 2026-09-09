export { MaterialXLoader } from './MaterialXLoader.js';
export type { MaterialXLoaderParseOptions } from './MaterialXLoader.js';

export { MaterialXDocument, MaterialXNode } from './MaterialXDocument.js';
export type { MaterialXParseResult } from './MaterialXDocument.js';

export { MaterialXLog, MaterialXLogCodes } from './MaterialXLog.js';
export type { MaterialXLogCode, MaterialXLogEntry } from './MaterialXLog.js';

export { createStrictInterfaceValidator } from './MaterialXInterfaceValidation.js';

export { isZipBuffer, readMtlxArchive, createArchiveResolver } from './MaterialXArchive.js';
export type { MaterialXArchiveResult } from './MaterialXArchive.js';

export {
	getSurfaceMapper,
	getSupportedSurfaceCategories,
	MaterialXSurfaceMappings,
} from './MaterialXSurfaceMappings.js';

export { MtlXLibrary } from './MaterialXNodeLibrary.js';
export type { MXElement } from './MaterialXNodeLibrary.js';

export { createMaterialXCompileRegistry, compileNodeFromRegistry } from './compile/MaterialXCompileRegistry.js';
export type { MaterialXCompileContext } from './compile/MaterialXCompileRegistry.js';

export { MaterialXLoader as default } from './MaterialXLoader.js';
