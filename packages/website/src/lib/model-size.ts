import {
  computeDecoderInputSize, computeGridLevels, computeMLPFlops,
  computeMLPParamCount, computeModelFootprint,
} from 'three-ntc-trainer';

export interface ModelSizeSettings {
  baseResolution: number;
  levels: number;
  hiddenSize: number;
  hiddenLayers: number;
  positionalEncoding: boolean;
  dualGrid: boolean;
  quantization: 'none' | 'uint8' | 'uint4' | 'uint2';
}

/** The website trains four-channel G0 and G1 grids. No model allocation needed. */
export function estimateModelSize(settings: ModelSizeSettings, outputChannels: number) {
  const resolutions = computeGridLevels(settings.baseResolution, settings.levels);
  const gridParams = resolutions.reduce((sum, r) =>
    sum + 4 * (r * r + (settings.dualGrid ? Math.max(1, Math.floor(r / 2)) ** 2 : 0)), 0);
  const inputSize = computeDecoderInputSize(4, settings.positionalEncoding, settings.dualGrid);
  const shape = { inputSize, hiddenSize: settings.hiddenSize, hiddenLayers: settings.hiddenLayers, outputSize: outputChannels };
  const mlpParams = computeMLPParamCount(shape);
  // NTCManifest exports uint8 even when QAT is off. Do not imply lossless storage.
  const gridStorageBits = settings.quantization === 'uint2' ? 2 : settings.quantization === 'uint4' ? 4 : 8;
  return {
    ...computeModelFootprint({ gridParams, mlpParams, flops: computeMLPFlops(shape), gridStorageBits }),
    gridParams, mlpParams, inputSize, gridStorageBits,
  };
}
