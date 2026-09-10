import { describe, expect, it } from 'vitest';

import { createNTCGridPyramidModel, computeDecoderInputSize } from './NTCGridPyramidModel.js';
import { computeTextureModelLayout } from './NTCGPUModel.js';
import { encodeNTC } from './NTCManifest.js';
import { NTCLoader } from 'three-ntc';

describe('NTCGridPyramidModel', () => {
  it('allocates independent half-resolution G1 grids and round-trips wide features', () => {
    const options={gridChannels:8,lowResChannels:12,levels:3,baseResolution:16,
      dualGrid:true,positionalEncoding:true,outputChannels:3,textureResolution:64};
    const model=createNTCGridPyramidModel(options,()=>0.6);
    expect(model.grids.map(g=>g.width)).toEqual([16,4,1]);
    expect(model.lowResGrids.map(g=>g.width)).toEqual([8,2,1]);
    expect(model.lowResGrids.every(g=>g.channels===12)).toBe(true);
    expect(model.lowResGrids[2].data).not.toBe(model.grids[2].data);
    const layout=computeTextureModelLayout(options);
    expect(layout.inputSize).toBe(4*8+12+12+1);
    expect(layout.totalLatents).toBe([...model.grids,...model.lowResGrids].reduce((s,g)=>s+g.data.length,0));
    const manifest=encodeNTC(model,{activeChannels:[{key:'albedo'}],constantValues:{}});
    const loaded=new NTCLoader().parse(manifest).cpuModel;
    expect(loaded.lowResGrids?.map(g=>[g.width,g.channels])).toEqual([[8,12],[2,12],[1,12]]);
    expect(loaded.decoder.layers[0].inputSize).toBe(layout.inputSize);
  });
  it('defaults to the plain bilinear-tap decoder input width', () => {
    expect(computeDecoderInputSize(4, false)).toBe(5); // channels + LOD
    expect(computeDecoderInputSize(2, false)).toBe(3);
  });

  it('widens to the paper-style 4-tap + positional-encoding input when enabled', () => {
    // 4 * channels (4-tap concat) + 12 (positional encoding) + 1 (LOD)
    expect(computeDecoderInputSize(4, true)).toBe(4 * 4 + 12 + 1);
    expect(computeDecoderInputSize(2, true)).toBe(4 * 2 + 12 + 1);
  });

  it('adds one bilinear G1 tap when dualGrid is on, in either mode', () => {
    expect(computeDecoderInputSize(4, false, true)).toBe(4 + 4 + 1);
    expect(computeDecoderInputSize(4, true, true)).toBe(4 * 4 + 12 + 4 + 1);
    const dual = createNTCGridPyramidModel({ channels: 4, outputChannels: 3, levels: 2, baseResolution: 8, dualGrid: true }, () => 0.5);
    expect(dual.dualGrid).toBe(true);
    expect(dual.decoder.layers[0].inputSize).toBe(9);
    expect(computeTextureModelLayout({ channels: 4, outputChannels: 3, levels: 2, baseResolution: 8, dualGrid: true }).inputSize).toBe(9);
  });

  it('sizes the trained decoder MLP to match, and carries the flag on the model', () => {
    const plain = createNTCGridPyramidModel({ channels: 4, outputChannels: 3, levels: 2, baseResolution: 8 }, () => 0.5);
    expect(plain.positionalEncoding).toBe(false);
    expect(plain.decoder.layers[0].inputSize).toBe(5);

    const withPE = createNTCGridPyramidModel(
      { channels: 4, outputChannels: 3, levels: 2, baseResolution: 8, positionalEncoding: true },
      () => 0.5,
    );
    expect(withPE.positionalEncoding).toBe(true);
    expect(withPE.decoder.layers[0].inputSize).toBe(4 * 4 + 12 + 1);
  });
});

it('physical mip limits match GPU textures for unit and non-power-of-two sources', () => {
  for(const [size,lod] of [[1,0],[3,1],[6,2],[1000,9]]) {
    const options={textureResolution:size,baseResolution:2,levels:1,hiddenSizes:[],outputChannels:3};
    const cpu=createNTCGridPyramidModel(options,()=>0.5);
    expect(cpu.maxLod).toBe(lod);
    expect(computeTextureModelLayout(options).maxLod).toBe(lod);
    const manifest=encodeNTC(cpu,{activeChannels:[{key:'albedo'}],constantValues:{}});
    expect(new NTCLoader().parse(manifest).cpuModel.maxLod).toBe(lod);
  }
});
