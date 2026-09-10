// Browser/WebGPU tests for the runtime decoder (NTCDecoderTSL.ts +
// NTCNodeMaterial.ts), checked pixel-by-pixel against the plain-JS CPU
// reference in /test/gpu-helpers.ts. Runs under the `gpu` vitest project
// (real Chromium + WebGPU, see /vitest.config.ts).
import { float, uv, vec4 } from 'three/tsl';
import { beforeAll, describe, expect, it } from 'vitest';

import { evaluateNeuralTextureRaw } from './NTCDecoderTSL.js';
import { buildLevelTextures, buildMipChainTexture } from './NTCHalfFloatTexture.js';
import { NTCLoader } from './NTCLoader.js';
import { NTCNodeMaterial } from './NTCNodeMaterial.js';
import {
  evaluateNTCCpu,
  getRenderer,
  makeModel,
  pixelUv,
  renderNodeToFloats,
} from '../../../test/gpu-helpers.js';

const SIZE = 32;

let renderer: any;
beforeAll(async () => {
  renderer = await getRenderer();
});

/** Renders the raw decoder outputs (first 4 channels) at a fixed LOD. */
async function renderDecoder(cpuModel: any, lod: number): Promise<Float32Array> {
  const mipChain = cpuModel.positionalEncoding ? null : buildMipChainTexture(cpuModel);
  const levelTextures = cpuModel.positionalEncoding ? buildLevelTextures(cpuModel) : null;
  const outputs = evaluateNeuralTextureRaw(
    uv(),
    cpuModel,
    mipChain,
    null,
    float(lod),
    levelTextures,
  );
  const node = vec4(outputs[0], outputs[1], outputs[2], outputs[3] ?? float(0));
  const pixels = await renderNodeToFloats(renderer, node, SIZE);
  mipChain?.dispose();
  levelTextures?.forEach((t: any) => t.dispose());
  return pixels;
}

describe('harness', () => {
  it('maps pixel (x, y) to uv (x+0.5, y+0.5)/size', async () => {
    const pixels = await renderNodeToFloats(renderer, vec4(uv(), 0, 1), 8);
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const [u, v] = pixelUv(8, x, y);
        expect(pixels[(y * 8 + x) * 4]).toBeCloseTo(u, 2);
        expect(pixels[(y * 8 + x) * 4 + 1]).toBeCloseTo(v, 2);
      }
    }
  });
});

describe('evaluateNeuralTextureRaw matches the CPU reference decoder', () => {
  it('preserves native alternating features inside a multi-mip band', async () => {
    const model = makeModel(1, { gridChannels: 1, levels: 2, baseResolution: 4,
      mipsPerLevel: 2, hiddenSizes: [], outputChannels: 4, textureResolution: 16 });
    model.grids[0].data.set(Array.from({length: 16}, (_, i) => i % 2));
    const layer = model.decoder.layers[0];
    layer.weights.fill(0);
    layer.biases.fill(0);
    layer.weights[0] = 1;
    const a = await renderDecoder(model, 0);
    const b = await renderDecoder(model, 1);
    expect(Math.max(...a) - Math.min(...a)).toBeGreaterThan(0.8);
    expect(Math.max(...a.map((v, i) => Math.abs(v - b[i])))).toBeLessThan(0.002);
  });
  const configs = [
    { positionalEncoding: false, dualGrid: false, hiddenActivation: 'relu', mipsPerLevel: 1 },
    { positionalEncoding: false, dualGrid: true, hiddenActivation: 'hgelu', mipsPerLevel: 2 },
    { positionalEncoding: true, dualGrid: false, hiddenActivation: 'relu', mipsPerLevel: 2 },
    { positionalEncoding: true, dualGrid: true, hiddenActivation: 'hgelu', mipsPerLevel: 1 },
  ];
  const lods = [0, 1, 2.5, 5];

  for (const config of configs) {
    for (const lod of lods) {
      it(`${JSON.stringify(config)} lod=${lod}`, async () => {
        const cpuModel = makeModel(7, {
          ...config,
          gridChannels: 4,
          levels: 3,
          baseResolution: 16,
          hiddenSizes: [16, 16],
          outputChannels: 4,
          textureResolution: 64,
        });
        const pixels = await renderDecoder(cpuModel, lod);

        let maxErr = 0;
        let sumErr = 0;
        for (let y = 0; y < SIZE; y++) {
          for (let x = 0; x < SIZE; x++) {
            const [u, v] = pixelUv(SIZE, x, y);
            const expected = evaluateNTCCpu(cpuModel, u, v, lod);
            for (let c = 0; c < 4; c++) {
              const err = Math.abs(pixels[(y * SIZE + x) * 4 + c] - expected[c]);
              maxErr = Math.max(maxErr, err);
              sumErr += err;
            }
          }
        }
        // Half-float render target + hardware bilinear weight precision.
        expect(maxErr).toBeLessThan(2e-2);
        expect(sumErr / (SIZE * SIZE * 4)).toBeLessThan(4e-3);
      });
    }
  }
});

describe('shipped .ntc assets', () => {
  const assets = import.meta.glob('../../website/public/ntc/*.ntc', {
    query: '?raw',
    import: 'default',
    eager: true,
  }) as Record<string, string>;

  for (const [path, json] of Object.entries(assets)) {
    it(`${path.split('/').pop()} decodes and renders on the GPU like the CPU reference`, async () => {
      const { cpuModel, channelClassification } = new NTCLoader().parse(json);

      // Raw decoder outputs vs CPU reference at LOD 0.
      const pixels = await renderDecoder(cpuModel, 0);
      let maxErr = 0;
      for (let y = 0; y < SIZE; y += 4) {
        for (let x = 0; x < SIZE; x += 4) {
          const [u, v] = pixelUv(SIZE, x, y);
          const expected = evaluateNTCCpu(cpuModel, u, v, 0);
          for (let c = 0; c < Math.min(4, cpuModel.outputChannels); c++) {
            maxErr = Math.max(maxErr, Math.abs(pixels[(y * SIZE + x) * 4 + c] - expected[c]));
          }
        }
      }
      // Trained models have larger-magnitude latents/weights than the random
      // ones above, so the half-float error budget is proportionally larger.
      expect(maxErr).toBeLessThan(5e-2);

      // The full material renders finite, in-range albedo.
      const material = new NTCNodeMaterial(cpuModel, channelClassification, {
        debugView: 'albedo',
        lodNode: float(0),
      });
      const albedo = await renderNodeToFloats(renderer, (material as any).colorNode, SIZE);
      material.dispose();
      for (let i = 0; i < albedo.length; i += 4) {
        for (let c = 0; c < 3; c++) {
          expect(albedo[i + c]).toBeGreaterThanOrEqual(0);
          expect(albedo[i + c]).toBeLessThanOrEqual(1);
        }
      }
    });
  }
});
