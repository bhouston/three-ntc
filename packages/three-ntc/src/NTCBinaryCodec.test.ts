// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import {
  base64FromBytes,
  bytesFromBase64,
  decodeFloat16Base64,
  decodeUint8Base64,
  encodeFloat16Base64,
  encodeUint8Base64,
  float16ToFloat32,
  float32ToFloat16,
} from './NTCBinaryCodec.js';

describe('NTCBinaryCodec', () => {
  it('round-trips byte arrays through base64', () => {
    const bytes = new Uint8Array(50000);
    for (let i = 0; i < bytes.length; i++) bytes[i] = i % 256;

    expect(Array.from(bytesFromBase64(base64FromBytes(bytes)))).toEqual(Array.from(bytes));
    expect(bytesFromBase64(base64FromBytes(new Uint8Array(0))).length).toBe(0);
  });

  it('round-trips float16 and uint8 payloads', () => {
    for (const value of [0, 1, -1, 0.5, -0.5, 2, 100, -100]) {
      expect(Math.abs(float16ToFloat32(float32ToFloat16(value)) - value)).toBeLessThanOrEqual(1e-5);
    }

    const floats = new Float32Array([0, 1, -1, 0.25, 3.5, -7.125, 0.001]);
    const decodedFloats = decodeFloat16Base64(encodeFloat16Base64(floats), floats.length);

    for (let i = 0; i < floats.length; i++) {
      expect(Math.abs(decodedFloats[i] - floats[i])).toBeLessThan(Math.max(1e-3, Math.abs(floats[i]) * 1e-2));
    }

    const min = -2;
    const max = 3;
    const values = new Float32Array([-2, -1, 0, 0.5, 1, 2.9, 3]);
    const decodedUint8 = decodeUint8Base64(encodeUint8Base64(values, min, max), min, max, values.length);
    const step = (max - min) / 255;

    for (let i = 0; i < values.length; i++) {
      expect(Math.abs(decodedUint8[i] - values[i])).toBeLessThanOrEqual(step / 2 + 1e-6);
    }

    expect(Array.from(decodeUint8Base64(encodeUint8Base64(new Float32Array([5, 5]), 5, 5), 5, 5, 2))).toEqual([5, 5]);
  });
});
