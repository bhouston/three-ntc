// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import {
  computeGridLevels,
  createLatentGrid,
  DEFAULT_MIPS_PER_LEVEL,
  LATENT_INIT_SCALE,
  MAX_GRID_RESOLUTION,
} from './NTCGridModel.js';

describe('NTCGridModel', () => {
  it('computes grid levels with stock mip spacing', () => {
    expect(computeGridLevels(128, 4, 2)).toEqual([128, 32, 8, 2]);
    expect(computeGridLevels(128, 4)).toEqual(computeGridLevels(128, 4, DEFAULT_MIPS_PER_LEVEL));
    expect(computeGridLevels(8, 6, 2)).toEqual([8, 2, 1, 1, 1, 1]);
    expect(computeGridLevels(MAX_GRID_RESOLUTION + 1000, 1, 2)).toEqual([MAX_GRID_RESOLUTION]);
  });

  it('creates latent grids with deterministic initialization', () => {
    let calls = 0;
    const grid = createLatentGrid(3, 2, 5, () => {
      calls++;
      return 0.5;
    });

    expect(grid.data.length).toBe(3 * 2 * 5);
    expect(grid.data).toBeInstanceOf(Float32Array);
    expect(Array.from(createLatentGrid(1, 1, 4, () => 0.5).data)).toEqual([0, 0, 0, 0]);
    expect(Math.abs(createLatentGrid(1, 1, 1, () => 0).data[0] + LATENT_INIT_SCALE)).toBeLessThanOrEqual(1e-7);
    expect(calls).toBe(3 * 2 * 5);
  });
});
