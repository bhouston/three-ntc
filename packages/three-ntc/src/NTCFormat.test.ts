// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { buildChannelActivations, CHANNELS, getChannel, layoutChannels, MAX_TOTAL_CHANNELS } from './NTCFormat.js';

describe('NTCFormat', () => {
  it('lays out channel subsets contiguously', () => {
    const subset = [getChannel('normal'), getChannel('albedo'), getChannel('roughness')];
    const layout = layoutChannels(subset);

    expect(layout.channels[0].key).toBe('normal');
    expect(layout.channels[0].offset).toBe(0);
    expect(layout.channels[1].offset).toBe(2);
    expect(layout.totalChannels).toBe(6);
    expect(layout.packCount).toBe(2);
    expect(layoutChannels([])).toEqual({ channels: [], totalChannels: 0, packCount: 0 });
    expect(MAX_TOTAL_CHANNELS).toBe(layoutChannels(CHANNELS).totalChannels);
  });

  it('builds output activations from channel metadata', () => {
    const { channels } = layoutChannels([getChannel('albedo'), getChannel('normal'), getChannel('roughness')]);

    expect(buildChannelActivations(channels)).toEqual(['sigmoid', 'sigmoid', 'sigmoid', 'tanh', 'tanh', 'sigmoid']);
    expect(() => getChannel('nonexistent')).toThrow(/unknown channel/);
  });
});
