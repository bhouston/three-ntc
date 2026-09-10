import { expect, it } from 'vitest';
import { createNTCGridPyramidModel, encodeNTC } from 'three-ntc-trainer';
import { estimateModelSize, type ModelSizeSettings } from './model-size.js';

const defaults: ModelSizeSettings = {
  baseResolution: 16, levels: 2, hiddenSize: 16, hiddenLayers: 1,
  positionalEncoding: true, dualGrid: true, quantization: 'uint8',
};

for (const positionalEncoding of [false, true]) {
  for (const dualGrid of [false, true]) {
    it.each(['none', 'uint8', 'uint4', 'uint2'] as const)(
      `matches the exported payload with PE=${positionalEncoding}, dual=${dualGrid}, quantization=%s`,
      quantization => {
        const settings = { ...defaults, positionalEncoding, dualGrid, quantization };
        const estimate = estimateModelSize(settings, 3);
        const model = createNTCGridPyramidModel({
          ...settings, channels: 4, outputChannels: 3, hiddenSizes: [settings.hiddenSize],
        }, () => 0.5);
        model.quantization = { mode: quantization };
        const manifest = encodeNTC(model, { activeChannels: [{key: 'albedo'}], constantValues: {} });
        const blocks = [...manifest.latents.levels, ...(manifest.latents.lowResLevels ?? []), manifest.mlp];
        const bytes = blocks.reduce((sum, block) => sum + Buffer.from(block.dataBase64, 'base64').byteLength, 0);
        const parameters = model.decoder.layers.reduce((sum, layer) => sum + layer.weights.length + layer.biases.length, 0);
        expect(estimate.storageBytes).toBe(bytes);
        expect(estimate.mlpParams).toBe(parameters);
        expect(estimate.inputSize).toBe(model.decoder.layers[0].inputSize);
        expect(estimate.memoryBytes).toBe(
          [...model.grids, ...model.lowResGrids].reduce((sum, grid) => sum + grid.data.length * 2, 0) + parameters * 4,
        );
      },
    );
  }
}

it('quantization changes packed storage, not the decoded runtime representation', () => {
  const sizes = ['uint8', 'uint4', 'uint2'].map(quantization => estimateModelSize({ ...defaults, quantization } as ModelSizeSettings, 3));
  expect(sizes.map(size => size.storageBytes)).toEqual([2550, 1870, 1530]);
  expect(new Set(sizes.map(size => size.memoryBytes)).size).toBe(1);
  expect(estimateModelSize({...defaults, quantization: 'none'}, 3)).toEqual(sizes[0]);
});
