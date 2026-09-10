import { constantEqualsDefault, getChannel, type NTCChannelClassification } from 'three-ntc';
import { computeGridLevels } from 'three-ntc-trainer';
import { estimateModelSize, type ModelSizeSettings } from './model-size.js';

export interface ModelInfoData {
  name: string;
  size: Pick<ReturnType<typeof estimateModelSize>, 'memoryBytes' | 'storageBytes' | 'mlpParams' | 'inputSize' | 'flopsPerDecode'>;
  grids: { group: 'G0' | 'G1'; width: number; height: number; channels: number; bits: number }[];
  layers: { inputSize: number; outputSize: number }[];
  activeChannels: string[];
  constantChannels: string[];
  trainingQuantizationOff?: boolean;
}

function channelLists(classification?: NTCChannelClassification | null) {
  const activeChannels = (classification?.activeChannels ?? []).map(channel => channel.key);
  const constantChannels = Object.entries(classification?.constantValues ?? {})
    .filter(([key, value]) => !activeChannels.includes(key) && !constantEqualsDefault(getChannel(key), value))
    .map(([key]) => key);
  return {activeChannels, constantChannels};
}

export function configuredModelInfo(settings: ModelSizeSettings, outputChannels: number, name: string, classification?: NTCChannelClassification | null): ModelInfoData {
  const size = estimateModelSize(settings, outputChannels);
  const resolutions = computeGridLevels(settings.baseResolution, settings.levels);
  const grids: ModelInfoData['grids'] = resolutions.map(r => ({group:'G0', width:r, height:r, channels:4, bits:size.gridStorageBits}));
  if (settings.dualGrid) grids.push(...resolutions.map(r => ({group:'G1' as const, width:Math.max(1, Math.floor(r/2)), height:Math.max(1, Math.floor(r/2)), channels:4, bits:size.gridStorageBits})));
  const widths = [size.inputSize, ...Array(settings.hiddenLayers).fill(settings.hiddenSize), outputChannels];
  return {name, size, grids, layers: widths.slice(1).map((outputSize,i) => ({inputSize:widths[i], outputSize})),
    ...channelLists(classification), trainingQuantizationOff:settings.quantization === 'none'};
}

/** Uses the loaded architecture and actual encoded blocks, including G1 and mixed bit depths. */
export function loadedModelInfo(manifest: any, cpuModel: any, classification: NTCChannelClassification, name: string): ModelInfoData {
  const blocks = [...manifest.latents.levels, ...(manifest.latents.lowResLevels ?? [])];
  const grids: ModelInfoData['grids'] = blocks.map((grid:any, i:number) => ({
    group:i < manifest.latents.levels.length ? 'G0' : 'G1', width:grid.width, height:grid.height,
    channels:grid.channels, bits:Number(grid.dtype.replace('uint','')),
  }));
  const layers = cpuModel.decoder.layers;
  const mlpParams = layers.reduce((sum:number, layer:any) => sum + layer.weights.length + layer.biases.length, 0);
  const flopsPerDecode = layers.reduce((sum:number, layer:any) => sum + 2*layer.inputSize*layer.outputSize + layer.outputSize, 0);
  return {name, grids, layers:layers.map((layer:any) => ({inputSize:layer.inputSize, outputSize:layer.outputSize})),
    size:{mlpParams, inputSize:layers[0].inputSize, flopsPerDecode,
      memoryBytes:grids.reduce((sum, grid) => sum + grid.width*grid.height*grid.channels*2, 0) + mlpParams*4,
      storageBytes:[...blocks, manifest.mlp].reduce((sum, block) => sum + atob(block.dataBase64).length, 0)},
    ...channelLists(classification)};
}
