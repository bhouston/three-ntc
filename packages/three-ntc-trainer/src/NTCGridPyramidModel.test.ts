import { describe, expect, it } from 'vitest';

import { createNTCGridPyramidModel, computeDecoderInputSize } from './NTCGridPyramidModel.js';
import { computeTextureModelLayout } from './NTCGPUModel.js';

describe('NTCGridPyramidModel', () => {
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
