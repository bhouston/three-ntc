// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { getNTCProfile, NTC_PROFILE_NAMES, NTC_PROFILES } from './NTCProfiles.js';

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
