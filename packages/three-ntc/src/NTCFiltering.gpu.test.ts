import { float, vec2, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { evaluateNeuralTextureFiltered, evaluateNeuralTextureSampled } from './NTCDecoderTSL.js';
import { buildLevelTextures } from './NTCHalfFloatTexture.js';
import { getRenderer, makeModel, renderNodeToFloats } from '../../../test/gpu-helpers.js';

for (const mipBlend of [false, true]) {
  it(`filters activated decoded values across ${mipBlend ? 'mips' : 'texels'}`, async () => {
    const renderer = await getRenderer();
    const model = makeModel(1, {gridChannels:1, levels:1, baseResolution:2,
      textureResolution:2, hiddenSizes:[], outputChannels:4});
    model.grids[0].data.set([-2,0,-2,0]);
    const layer = model.decoder.layers[0];
    layer.weights.fill(0); layer.biases.fill(mipBlend ? -2 : 0);
    for(let c=0;c<4;c++) layer.weights[c*layer.inputSize+(mipBlend ? layer.inputSize-1 : 0)] = mipBlend ? 6 : 1;
    const textures = buildLevelTextures(model);
    const sigmoid = (x:number) => 1/(1+Math.exp(-x));
    const values = evaluateNeuralTextureFiltered(vec2(0.5,0.25),model,textures,
      float(mipBlend ? 0.5 : 0),['sigmoid','sigmoid','sigmoid','sigmoid']);
    const result = await renderNodeToFloats(renderer,vec4(...values),1);
    const expected = (sigmoid(-2)+sigmoid(mipBlend ? 4 : 0))/2;
    for (const v of result) expect(Math.abs(v-expected)).toBeLessThan(0.001);
    // Applying the activation after filtering would produce a different result.
    expect(Math.abs(expected-sigmoid(mipBlend ? 1 : -1))).toBeGreaterThan(0.03);
    const nearest = evaluateNeuralTextureSampled(vec2(0.5,0.25),model,textures,
      float(mipBlend ? 0.5 : 0),['sigmoid','sigmoid','sigmoid','sigmoid'],float(0));
    const raw = await renderNodeToFloats(renderer,vec4(...nearest),1);
    for(const v of raw) expect(Math.abs(v-sigmoid(mipBlend ? 4 : 0))).toBeLessThan(0.001);
    textures.forEach(t=>t.dispose());
  });
}
