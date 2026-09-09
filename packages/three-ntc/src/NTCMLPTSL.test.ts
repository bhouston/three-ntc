// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { Matrix4 } from 'three';
import { describe, expect, it } from 'vitest';

import { packLayerBiasesVec4, packLayerWeightsMat4 } from './NTCMLPTSL.js';

describe('NTCMLPTSL', () => {
  it('packs weights and biases into fp32 mat4/vec4 blocks', () => {
    const inputSize = 6;
    const outputSize = 5;
    const weights = new Array(outputSize * inputSize);

    for (let o = 0; o < outputSize; o++) {
      for (let i = 0; i < inputSize; i++) weights[o * inputSize + i] = o * inputSize + i + 1;
    }

    const packed = packLayerWeightsMat4(weights, inputSize, outputSize);

    expect(packed.length).toBe(4); // ceil(output/4) * ceil(input/4) mat4 blocks
    expect(packed[0]).toBeInstanceOf(Matrix4);
    expect(packed[0].equals(new Matrix4().set(1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22))).toBe(true);
    expect(packed[3].equals(new Matrix4().set(29, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))).toBe(true);

    const biases = packLayerBiasesVec4([1, 2, 3, 4, 5]);
    expect([biases[0].x, biases[0].y, biases[0].z, biases[0].w]).toEqual([1, 2, 3, 4]);
    expect([biases[1].x, biases[1].y, biases[1].z, biases[1].w]).toEqual([5, 0, 0, 0]);
  });
});
