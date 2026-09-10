// Public API of three-ntc-trainer: a GPU trainer that bakes a three.js
// material into a Neural Texture Compression (.ntc) model.

export { NTCTrainer } from './NTCTrainer.js';
export type { NTCTrainerOptions, NTCTrainProgress, NTCTrainArgs } from './NTCTrainer.js';

export { fitNTCMaterial } from './NTCFit.js';

export { NTCExporter } from './NTCExporter.js';
export type { NTCExporterOptions } from './NTCExporter.js';

export { encodeNTC, FORMAT, VERSION } from './NTCManifest.js';

export {
	bakeMaterialToTextures,
	buildPackedColorNodes,
	buildChannelPreviewMaterials,
	classifyMaterialChannels,
	resolveMaterialChannelNodes
} from './NTCSource.js';

export { NTCTextureSource, bakeColorNodeToTexture, extractBaseColorNode, loadImageTexture } from './NTCTextureSource.js';

export { NTC_PROFILES, NTC_PROFILE_NAMES, getNTCProfileControls, getNTCPaperProfile, getNTCProfile } from './NTCProfiles.js';
export type { NTCProfile } from './NTCProfiles.js';

export { computeModelFootprint, formatModelSizeSummary, formatBytes, formatFlops, computeMLPParamCount, computeMLPFlops, computeGridLatentTexels, computeMLPLayoutStats } from './NTCModelSize.js';

export { NTCLossGraph } from './NTCLossGraph.js';
export type { NTCLossSeries, NTCLossPoint } from './NTCLossGraph.js';

export { MATERIALX_SAMPLES, getMaterialXSample, getMaterialXSampleUrl, populateMaterialXSelect } from './NTCMaterialXSamples.js';

export { inferAlbedoUvTransform, inferUvTransformFromImageNode, findImageNode } from './NTCMaterialXUvTransform.js';

// Grid/MLP shape building blocks - useful for a trainer UI's GUI dropdowns
// (see NTCGridModel.js's option-list constants) and for advanced callers
// building a model by hand rather than through NTCTrainer/fitNTCMaterial.
export {
	computeGridLevels,
	GRID_LEVELS_OPTIONS,
	GRID_BASE_RESOLUTION_OPTIONS,
	MLP_HIDDEN_SIZE_OPTIONS,
	MLP_ACTIVATION_OPTIONS,
	MAX_GRID_RESOLUTION,
	DEFAULT_MIPS_PER_LEVEL
} from './NTCGridModel.js';

export { createNTCGridPyramidModel, resolveNTCGridPyramidOptions, computeDecoderInputSize } from './NTCGridPyramidModel.js';
export type { NTCGridPyramidModel, NTCGridPyramidOptions } from './NTCGridPyramidModel.js';

export { NTCGPUModel, computeTextureModelLayout } from './NTCGPUModel.js';
export type { NTCGPUModelOptions, NTCTextureModelLayout } from './NTCGPUModel.js';

export { DEFAULT_QUANTIZATION_OPTIONS, resolveQuantizationConfig, computeLatentRanges } from './NTCQuantization.js';
export type { NTCQuantizationOptions, ResolvedNTCQuantizationConfig } from './NTCQuantization.js';

export { getLearningRate, createRandom, yieldToBrowser } from './NTCTrainingUtils.js';

// Re-exported once the sibling MaterialX loader port (a parallel effort)
// lands its `MaterialXLoader.ts` file under `./materialx/`.
export { MaterialXLoader } from './materialx/MaterialXLoader.js';
