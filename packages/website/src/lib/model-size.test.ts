import { expect, it } from 'vitest';
import { createNTCGridPyramidModel, encodeNTC } from 'three-ntc-trainer';
import { estimateModelSize, type ModelSizeSettings } from './model-size.js';

const defaults: ModelSizeSettings = {
  baseResolution: 16, levels: 2, hiddenSize: 16, hiddenLayers: 1,
  positionalEncoding: true, dualGrid: true, quantization: 'uint8', samplingMode: 'nearest',
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


it.each(['nearest', 'stochastic', 'trilinear'] as const)('counts all MLP evaluations for %s without changing model size', samplingMode => {
  const size = estimateModelSize({...defaults, samplingMode}, 3);
  // Fixture: 33 inputs -> 16 hidden -> 3 outputs. Two FLOPs per weight plus each bias.
  const singleDecode = 2 * (33 * 16 + 16 * 3) + 16 + 3;
  expect(size.flopsPerDecode).toBe(1171);
  expect(size.decoderEvaluations).toBe(samplingMode === 'trilinear' ? 8 : 1);
  expect(size.flops).toBe(singleDecode * (samplingMode === 'trilinear' ? 8 : 1));
  const nearest = estimateModelSize(defaults, 3);
  expect(size.storageBytes).toBe(nearest.storageBytes);
  expect(size.memoryBytes).toBe(nearest.memoryBytes);
  expect(size.mlpParams).toBe(nearest.mlpParams);
});
