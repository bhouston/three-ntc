import { expect, it } from 'vitest';
import { CHANNELS, getChannel, layoutChannels, NTCLoader } from 'three-ntc';
import { createNTCGridPyramidModel, encodeNTC } from 'three-ntc-trainer';
import { configuredModelInfo, loadedModelInfo } from './model-info.js';
import type { ModelSizeSettings } from './model-size.js';

it('reports matching model details before and after export, including both grids and active-only channel count', () => {
  const settings: ModelSizeSettings = {baseResolution:16, levels:2, hiddenSize:16, hiddenLayers:1,
    positionalEncoding:true, dualGrid:true, quantization:'uint4', samplingMode:'nearest'};
  const classification = {activeChannels:layoutChannels([getChannel('albedo')]).channels,
    constantValues:{...Object.fromEntries(CHANNELS.map(channel=>[channel.key,channel.defaultValue])), metalness:0.7}};
  const model = createNTCGridPyramidModel({...settings, channels:4, outputChannels:3, hiddenSizes:[16]}, () => 0.5);
  model.quantization = {mode:'uint4'};
  const manifest = encodeNTC(model, classification, {name:'Fixture'});
  const decoded = new NTCLoader().parse(manifest);
  const configured = configuredModelInfo(settings, 3, 'Fixture', classification);
  const loaded = loadedModelInfo(manifest, decoded.cpuModel, decoded.channelClassification, 'Fixture');
  expect(configured.size).toMatchObject(loaded.size);
  expect(configured.grids).toEqual(loaded.grids);
  expect(configured.layers).toEqual(loaded.layers);
  expect(loaded.grids.map(grid=>grid.group)).toEqual(['G0','G0','G1','G1']);
  expect(loaded.activeChannels).toEqual(['albedo']);
  expect(loaded.constantChannels).toEqual(['metalness']);
  expect(configured.activeChannels).toEqual(loaded.activeChannels);
  expect(configured.constantChannels).toEqual(loaded.constantChannels);
});
