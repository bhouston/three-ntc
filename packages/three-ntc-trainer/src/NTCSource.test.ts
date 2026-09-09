// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { classifyMaterialChannels } from './NTCSource.js';
import { CHANNELS } from 'three-ntc';

describe('NTCSource', () => {
  it('classifies active and constant channels', () => {
    const material = { colorNode: {}, roughness: 0.4, side: 2, transparent: true };
    const result = classifyMaterialChannels(material as any);

    expect(result.activeChannels.map((channel: any) => channel.key)).toEqual(['albedo']);
    expect(result.constantValues.roughness).toBe(0.4);
    expect(result.renderFlags).toEqual({ side: 2, transparent: true });

    const empty = classifyMaterialChannels({} as any);
    expect(empty.activeChannels.length).toBe(0);
    expect(Object.keys(empty.constantValues).length).toBe(CHANNELS.length);
    expect(empty.constantValues.normal).toEqual([0, 0, 1]);
  });
});
