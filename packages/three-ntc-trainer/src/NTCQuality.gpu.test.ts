// A measurement, not a quality threshold. Keep fixtures, budgets and seeds
// fixed so each implementation commit can be compared with its predecessor.
import { float, floor, sin, textureLevel, uv, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { NTCLoader } from 'three-ntc';
import { NTCTrainer } from './NTCTrainer.js';
import { encodeNTC } from './NTCManifest.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { buildLevelTextures, buildMipChainTexture } from '../../three-ntc/src/NTCHalfFloatTexture.js';
import { evaluateNeuralTextureRaw } from '../../three-ntc/src/NTCDecoderTSL.js';
import { getRenderer, renderNodeToFloats } from '../../../test/gpu-helpers.js';

for (const fixture of ['smooth', 'checker', 'waves']) {
  for (const positionalEncoding of [false, true]) {
    it(`measures ${fixture}, positionalEncoding=${positionalEncoding}`, async () => {
      const renderer = await getRenderer();
      const u = uv();
      const pattern = fixture === 'checker'
        ? floor(u.x.mul(8)).add(floor(u.y.mul(8))).mod(2)
        : sin(u.x.mul(fixture === 'waves' ? 100 : 6.283)).mul(sin(u.y.mul(25))).mul(0.4).add(0.5);
      // RGB albedo plus roughness; all four physical channels have range 1.
      const source = await bakeColorNodeToTexture(renderer,
        vec4(fixture === 'smooth' ? u.x : pattern, u.y, pattern, pattern.mul(0.7).add(0.15)),
        64, { generateMipmaps: true });
      const options = { gridChannels: 4, levels: 3, baseResolution: 16, mipsPerLevel: 2,
        hiddenSizes: [16,16], hiddenActivation: 'hgelu', outputChannels: 4,
        positionalEncoding, dualGrid: true, batchSize: 2048, iterations: 420,
        seed: 7, quantization: { mode: 'uint4' } };
      const result = await new NTCTrainer(options).train({ renderer, sourceTexture: source.texture });
      const manifest = encodeNTC(result.cpuModel, {activeChannels: [{key:'albedo'}, {key:'roughness'}], constantValues:{}});
      const loaded = new NTCLoader().parse(manifest).cpuModel;
      const metrics: Record<string, unknown> = {};
      for (const [label, model] of [['trained', result.cpuModel], ['exported', loaded]] as const) {
        const mip = model.positionalEncoding ? null : buildMipChainTexture(model);
        const levels = buildLevelTextures(model);
        let sum = 0, count = 0;
        const perMip: number[] = [];
        for (let lod = 0; lod <= 6; lod++) {
          const size = Math.max(1, 64 >> lod);
          const reference = await renderNodeToFloats(renderer, textureLevel(source.texture, uv(), lod), size);
          const values = evaluateNeuralTextureRaw(uv(), model, mip, null, float(lod), levels);
          const decoded = await renderNodeToFloats(renderer, vec4(...values), size);
          let error = 0;
          for (let i = 0; i < reference.length; i++) error += (reference[i] - decoded[i]) ** 2;
          perMip.push(error / reference.length);
          sum += error; count += reference.length;
        }
        metrics[label] = { mse: sum / count, psnr: -10 * Math.log10(sum / count), perMipMse: perMip };
        expect(Number.isFinite(sum)).toBe(true);
        mip?.dispose(); levels.forEach(t => t.dispose());
      }
      await (commands as any).recordMetric({fixture, positionalEncoding, seed:7,
        iterations:result.iterations, batchSize:2048, sourceSize:64, ...metrics});
      source.dispose();
    });
  }
}
