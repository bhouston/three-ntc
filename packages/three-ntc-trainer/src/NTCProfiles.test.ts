// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { getNTCProfileControls, getNTCPaperProfile, getNTCProfile, NTC_PROFILE_NAMES, NTC_PROFILES } from './NTCProfiles.js';

describe('NTCProfiles', () => {
  it('exposes valid named profiles', () => {
    expect(NTC_PROFILE_NAMES).toEqual(['mobile-fast', 'mobile-balanced', 'desktop-quality']);
    expect(Object.keys(NTC_PROFILES)).toEqual(NTC_PROFILE_NAMES);
    expect(getNTCProfile('not-a-real-profile')).toBeNull();

    const options = getNTCProfile('desktop-quality')!;
    options.hiddenSizes.push(999);
    expect(NTC_PROFILES['desktop-quality'].hiddenSizes.includes(999)).toBe(false);

    for (const name of NTC_PROFILE_NAMES) {
      const profile = NTC_PROFILES[name];
      const model = createNTCGridPyramidModel({ channels: 4, outputChannels: 3, ...getNTCProfile(name) } as any, () => 0.5);

      expect(model.hiddenSizes).toEqual(profile.hiddenSizes);
      expect(model.hiddenActivation).toBe(profile.hiddenActivation);
      expect(model.resolutions[0]).toBe(profile.baseResolution);
    }
  });
});

it('UI defaults and preset selection preserve both decoder width and depth', () => {
  for(const name of NTC_PROFILE_NAMES) {
    const controls=getNTCProfileControls(name)!;
    expect(Array(controls.hiddenLayers).fill(controls.hiddenSize)).toEqual(getNTCProfile(name)!.hiddenSizes);
  }
  expect(getNTCProfileControls('mobile-balanced')).toMatchObject({hiddenSize:32,hiddenLayers:2,levels:4});
  expect(getNTCProfileControls('mobile-fast')).toMatchObject({hiddenSize:16,hiddenLayers:2});
});

it('offers the paper architecture and explicitly expensive training budget', () => {
  const options=getNTCPaperProfile(1024);
  const model=createNTCGridPyramidModel(options,()=>0.5);
  expect(model.grids.map(g=>g.width)).toEqual([256,64,16,4]);
  expect(model.lowResGrids.map(g=>[g.width,g.channels])).toEqual([[128,12],[32,12],[8,12],[2,12]]);
  expect(model.decoder.layers.map(l=>[l.inputSize,l.outputSize,l.activation])).toEqual([[57,64,'hgelu'],[64,64,'hgelu'],[64,3,'linear']]);
  expect(options.batchSize*options.iterations).toBe(131072000000);
  expect(()=>getNTCPaperProfile(1000)).toThrow();
});
