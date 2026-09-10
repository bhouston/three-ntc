// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { DEFAULT_QUANTIZATION_OPTIONS, QUANTIZATION_SCHEMES, resolveQuantizationConfig } from './NTCQuantization.js';

describe('NTCQuantization', () => {
  it('quantizes CPU values and resolves config', () => {
    const lo = -2;
    const hi = 3;
    const step = (hi - lo) / 255;

    for (const value of [-2, -1, 0, 0.37, 1, 2.9, 3]) {
      const quantized = QUANTIZATION_SCHEMES.uint8.quantizeForwardCPU(value, lo, hi);
      expect(quantized).toBeGreaterThanOrEqual(lo - 1e-9);
      expect(quantized).toBeLessThanOrEqual(hi + 1e-9);
      expect(Math.abs(quantized - value)).toBeLessThanOrEqual(step / 2 + 1e-9);
    }

    expect(QUANTIZATION_SCHEMES.none.quantizeForwardCPU(0.123, -1, 1)).toBe(0.123);
    expect(resolveQuantizationConfig()).toEqual(DEFAULT_QUANTIZATION_OPTIONS);
    expect(resolveQuantizationConfig({ quantization: { mode: 'uint8', range: [-1, 1] } })).toEqual({
      mode: 'uint8',
      method: 'ste',
      target: 'latents',
      range: [-1, 1],
      perLevel: true,
    });
    expect(() => resolveQuantizationConfig({ quantization: { mode: 'int4' } })).toThrow(/quantization\.mode/);
    expect(() => resolveQuantizationConfig({ quantization: { range: [1, 0] } })).toThrow(/quantization\.range/);
  });
});

it('default noise ranges contain zero exactly at every supported bit depth', () => {
  for(const mode of ['uint2','uint4','uint8']) {
    const config=resolveQuantizationConfig({quantization:{mode}});
    const [lo,hi]=config.range as [number,number];
    expect(config.method).toBe('noise');
    expect(QUANTIZATION_SCHEMES[mode].quantizeForwardCPU(0,lo,hi)).toBe(0);
    expect(hi-lo).toBe(1-1/(2 ** Number(mode.slice(4))));
  }
  expect(()=>resolveQuantizationConfig({quantization:{mode:'uint4',method:'noise',range:'auto'}})).toThrow();
});
