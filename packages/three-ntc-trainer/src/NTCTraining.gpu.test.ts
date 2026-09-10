// Browser/WebGPU tests for the training kernels (NTCGPUComputeTSL.ts /
// NTCGPUKernelsTSL.ts) and NTCTrainer. Runs under the `gpu` vitest project
// (real Chromium + WebGPU, see /vitest.config.ts).
//
// The kernel's per-sample UV/LOD come from a GPU `sin()`-based hash that a
// CPU can't reproduce bit-for-bit, so instead of re-deriving samples the
// tests read the kernel's own decoder-input vector (`a0`) back and replay
// the MLP forward/backward on the CPU from there; the latent-gradient
// scatter (the one piece that depends on the sample UV) is checked by
// finite differences of the kernel's own loss.
import { float, uv, vec2, vec4 } from 'three/tsl';
import { beforeAll, describe, expect, it } from 'vitest';

import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { NTCTrainer } from './NTCTrainer.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { QUANTIZATION_SCHEMES } from './NTCQuantization.js';
import { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE } from './NTCGPUTrainingConstants.js';
import { createResetGradientNormComputeNode } from './NTCGPUKernelsTSL.js';
import {
  createAccumulateGradientNormComputeNode,
  createTextureAdamLatentsComputeNode,
  createTextureAdamWeightsComputeNode,
  createTextureTrainBatchComputeNode,
} from './NTCGPUComputeTSL.js';
import {
  activateDerivative,
  channelActivate,
  channelActivateDerivativeFromOutput,
  forwardCpu,
  getRenderer,
  maxAbsDiff,
} from '../../../test/gpu-helpers.js';

let renderer: any;
beforeAll(async () => {
  renderer = await getRenderer();
});

// Exactly representable in the half-float render target the bake lands in
// (and therefore unchanged by its mip chain), so the kernel's target is
// bit-identical to what the CPU replay uses.
const TARGET = [0.25, -0.5, 0.75, 0.625];
const CHANNEL_ACTIVATIONS = ['sigmoid', 'tanh', 'softplus', undefined];

async function constantTexture(size = 32) {
  const rt = await bakeColorNodeToTexture(renderer, vec4(...TARGET), size, {
    generateMipmaps: true,
  });
  return rt.texture;
}

/** Builds a CPU model + matching GPU model + the four training kernels, ready to step. */
function setup(options: Record<string, unknown>, seed = 3) {
  const cpuModel = createNTCGridPyramidModel(options as any, createRandom(seed));
  const gpuModel = new NTCGPUModel(options as any);
  gpuModel.initFromCPUModel(cpuModel);
  return { cpuModel, gpuModel };
}

async function readF32(attribute: any): Promise<Float32Array> {
  return new Float32Array(await renderer.getArrayBufferAsync(attribute));
}
async function readGrad(attribute: any): Promise<Float32Array> {
  const raw = new Int32Array(await renderer.getArrayBufferAsync(attribute));
  return Float32Array.from(raw, (v) => v / FIXED_POINT_SCALE);
}

describe('train-batch kernel', () => {
  it('interpolates quantized stored taps without re-quantizing the result', async () => {
    const texture = await constantTexture();
    const {gpuModel} = setup({gridChannels:1, levels:1, baseResolution:2,
      hiddenSizes:[4], outputChannels:4, dualGrid:true, batchSize:1,
      quantization:{mode:'uint2', range:[0,1]}});
    gpuModel.latentsBuffers.attribute.array.set([0,1,0,1]);
    gpuModel.latentsBuffers.attribute.needsUpdate = true;
    renderer.compute(createTextureTrainBatchComputeNode(gpuModel,[texture],
      {uv:vec2(0.45,0.5), lod:float(0)}));
    const act = await readF32(gpuModel.activationsAttribute);
    expect(act[0]).toBeCloseTo(0.4,5);
    expect(act[1]).toBeCloseTo(0.4,5); // G1 must use the same ordering.
    texture.dispose(); gpuModel.dispose();
  });
  const options = {
    gridChannels: 4,
    levels: 2,
    baseResolution: 8,
    mipsPerLevel: 2,
    hiddenSizes: [8, 8],
    hiddenActivation: 'hgelu',
    outputChannels: 4,
    channelActivations: CHANNEL_ACTIVATIONS,
    batchSize: 256,
    textureResolution: 32,
  };

  it('forward, loss, backward and weight gradients replay exactly on the CPU', async () => {
    const texture = await constantTexture();
    const { cpuModel, gpuModel } = setup(options);
    const { layout, batchSize } = gpuModel;
    const trainBatch = createTextureTrainBatchComputeNode(gpuModel, [texture]);
    gpuModel.stepUniform.value = 5;
    gpuModel.resetLoss();
    renderer.compute(trainBatch);

    const act = await readF32(gpuModel.activationsAttribute);
    const gpuLoss = await gpuModel.readLoss(renderer);
    const gpuGradW = await readGrad(gpuModel.weightsBuffers.gradAttribute);
    const layers = cpuModel.decoder.layers;

    const cpuGradW = new Float64Array(layout.totalWeights);
    let cpuLoss = 0;
    let maxActErr = 0;
    let maxDeltaErr = 0;

    for (let s = 0; s < batchSize; s++) {
      const base = s * layout.activationStride;
      const a0 = Array.from(
        act.slice(base + layout.a0Offset, base + layout.a0Offset + layout.inputSize),
      );
      const lodScalar = a0[layout.inputSize - 1];
      expect(lodScalar).toBeGreaterThanOrEqual(0);
      expect(lodScalar).toBeLessThanOrEqual(1);
      expect(
        Number.isInteger(
          lodScalar * layout.maxLod + 1e-6 - ((lodScalar * layout.maxLod + 1e-6) % 1),
        ),
      ).toBe(true);

      const { z, a } = forwardCpu(cpuModel.decoder, a0);
      for (let l = 0; l < layers.length; l++) {
        const zGpu = act.slice(
          base + layout.layerActs[l].zOffset,
          base + layout.layerActs[l].zOffset + layers[l].outputSize,
        );
        maxActErr = Math.max(maxActErr, maxAbsDiff(zGpu, z[l]));
        if (layout.layerActs[l].aOffset >= 0) {
          const aGpu = act.slice(
            base + layout.layerActs[l].aOffset,
            base + layout.layerActs[l].aOffset + layers[l].outputSize,
          );
          maxActErr = Math.max(maxActErr, maxAbsDiff(aGpu, a[l + 1]));
        }
      }

      // Output delta + loss.
      const last = layers.length - 1;
      let delta = z[last].map((zc, c) => {
        const pred = channelActivate(zc, CHANNEL_ACTIVATIONS[c]);
        const diff = pred - TARGET[c];
        cpuLoss += 0.5 * diff * diff;
        return diff * channelActivateDerivativeFromOutput(pred, CHANNEL_ACTIVATIONS[c]);
      });

      // Backward through the layers, accumulating weight/bias gradients.
      for (let l = last; l >= 0; l--) {
        const layer = layers[l];
        const dGpu = act.slice(
          base + layout.deltaOffsets[l],
          base + layout.deltaOffsets[l] + layer.outputSize,
        );
        maxDeltaErr = Math.max(maxDeltaErr, maxAbsDiff(dGpu, delta));
        const input = a[l];
        const { weightsOffset, biasesOffset } = layout.mlpLayers[l];
        for (let j = 0; j < layer.outputSize; j++) {
          cpuGradW[biasesOffset + j] += delta[j];
          for (let i = 0; i < layer.inputSize; i++)
            cpuGradW[weightsOffset + j * layer.inputSize + i] += delta[j] * input[i];
        }
        const prev: number[] = Array.from({ length: layer.inputSize }, () => 0);
        for (let i = 0; i < layer.inputSize; i++) {
          for (let j = 0; j < layer.outputSize; j++)
            prev[i] += delta[j] * layer.weights[j * layer.inputSize + i];
          if (l > 0) prev[i] *= activateDerivative(z[l - 1][i], layers[l - 1].activation);
        }
        if (l === 0) {
          const gradA0Gpu = act.slice(
            base + layout.gradA0Offset,
            base + layout.gradA0Offset + layer.inputSize,
          );
          maxDeltaErr = Math.max(maxDeltaErr, maxAbsDiff(gradA0Gpu, prev));
        }
        delta = prev;
      }
    }

    expect(maxActErr).toBeLessThan(2e-5);
    expect(maxDeltaErr).toBeLessThan(2e-5);
    // Fixed-point accumulation truncates each per-sample contribution to 1/FIXED_POINT_SCALE.
    const fixedPointBudget = batchSize / FIXED_POINT_SCALE;
    expect(Math.abs(gpuLoss - cpuLoss / batchSize)).toBeLessThan(
      fixedPointBudget / batchSize + 1e-5,
    );
    expect(maxAbsDiff(gpuGradW, cpuGradW)).toBeLessThan(fixedPointBudget + 1e-3);

    texture.dispose();
    gpuModel.dispose();
  });

  for (const variant of [
    { positionalEncoding: false, dualGrid: false },
    { positionalEncoding: true, dualGrid: true },
  ]) {
    it(`latent gradients match finite differences of the kernel loss (${JSON.stringify(variant)})`, async () => {
      const texture = await constantTexture(16);
      const { gpuModel } = setup({
        ...variant,
        gridChannels: 2,
        levels: 2,
        baseResolution: 4,
        mipsPerLevel: 1,
        hiddenSizes: [8],
        hiddenActivation: 'relu',
        outputChannels: 3,
        batchSize: 1024,
        textureResolution: 16,
      });
      const trainBatch = createTextureTrainBatchComputeNode(gpuModel, [texture]);
      gpuModel.stepUniform.value = 11;

      const lossSum = async () => {
        gpuModel.resetLoss();
        renderer.compute(trainBatch);
        return (await gpuModel.readLoss(renderer)) * gpuModel.batchSize;
      };

      await lossSum();
      const gradLatents = await readGrad(gpuModel.latentsBuffers.gradAttribute);
      const gradWeights = await readGrad(gpuModel.weightsBuffers.gradAttribute);

      const check = async (buffers: any, grad: Float32Array, indices: number[]) => {
        const values = buffers.attribute.array as Float32Array;
        for (const i of indices) {
          const h = 1e-2;
          const original = values[i];
          values[i] = original + h;
          buffers.attribute.needsUpdate = true;
          const plus = await lossSum();
          values[i] = original - h;
          buffers.attribute.needsUpdate = true;
          const minus = await lossSum();
          values[i] = original;
          buffers.attribute.needsUpdate = true;
          const fd = (plus - minus) / (2 * h);
          expect(Math.abs(fd - grad[i])).toBeLessThan(0.05 * Math.abs(grad[i]) + 0.02);
        }
      };

      const largest = (grad: Float32Array, n: number) =>
        Array.from(grad.keys())
          .sort((a, b) => Math.abs(grad[b]) - Math.abs(grad[a]))
          .slice(0, n);
      // Every latent (both levels) should receive gradient somewhere.
      expect(largest(gradLatents, 1).map((i) => Math.abs(gradLatents[i]))[0]).toBeGreaterThan(0.1);
      await check(gpuModel.latentsBuffers, gradLatents, [
        ...largest(gradLatents, 3),
        0,
        gradLatents.length - 1,
      ]);
      await check(gpuModel.weightsBuffers, gradWeights, largest(gradWeights, 2));

      texture.dispose();
      gpuModel.dispose();
    });
  }

  it('gradient-norm clipping and the first Adam step match the closed form', async () => {
    const texture = await constantTexture();
    const { gpuModel } = setup({ ...options, hiddenActivation: 'relu' });
    const trainBatch = createTextureTrainBatchComputeNode(gpuModel, [texture]);
    const resetNorm = createResetGradientNormComputeNode(gpuModel);
    const accumulateNorm = createAccumulateGradientNormComputeNode(gpuModel);
    const adamWeights = createTextureAdamWeightsComputeNode(gpuModel);
    const adamLatents = createTextureAdamLatentsComputeNode(gpuModel);
    const lr = 0.01;
    const maxNorm = 0.5;
    gpuModel.stepUniform.value = 1;
    gpuModel.learningRateUniform.value = lr;
    gpuModel.maxGradientNormUniform.value = maxNorm;

    gpuModel.resetLoss();
    renderer.compute(trainBatch);
    renderer.compute(resetNorm);
    renderer.compute(accumulateNorm);

    const invBatch = 1 / gpuModel.batchSize;
    const gW = (await readGrad(gpuModel.weightsBuffers.gradAttribute)).map((g) => g * invBatch);
    const gL = (await readGrad(gpuModel.latentsBuffers.gradAttribute)).map((g) => g * invBatch);
    const before = {
      w: Float32Array.from(gpuModel.weightsBuffers.attribute.array as Float32Array),
      l: Float32Array.from(gpuModel.latentsBuffers.attribute.array as Float32Array),
    };
    let normSq = 0;
    for (const g of gW) normSq += g * g;
    for (const g of gL) normSq += g * g;
    const gpuNormSq =
      new Int32Array(await renderer.getArrayBufferAsync(gpuModel.gradNormAttribute))[0] /
      GRADIENT_NORM_SCALE;
    expect(Math.abs(gpuNormSq - normSq)).toBeLessThan(
      1e-3 * normSq + (gW.length + gL.length) / GRADIENT_NORM_SCALE,
    );
    // Make sure the clip actually engages in this test.
    expect(Math.sqrt(normSq)).toBeGreaterThan(maxNorm);
    const clip = Math.min(1, maxNorm / Math.sqrt(normSq));

    renderer.compute(adamWeights);
    renderer.compute(adamLatents);

    // Step 1: mHat = g, vHat = g^2  =>  value -= lr * g / (|g| + eps).
    const expectAdam = async (buffers: any, grads: Float32Array, old: Float32Array) => {
      const values = await readF32(buffers.attribute);
      const m = await readF32(buffers.mAttribute);
      const v = await readF32(buffers.vAttribute);
      const gradAfter = await readGrad(buffers.gradAttribute);
      let maxErr = 0;
      for (let i = 0; i < values.length; i++) {
        const g = grads[i] * clip;
        const expected = old[i] - (lr * g) / (Math.abs(g) + 1e-7);
        maxErr = Math.max(
          maxErr,
          Math.abs(values[i] - expected),
          Math.abs(m[i] - 0.1 * g),
          Math.abs(v[i] - 0.001 * g * g),
        );
        expect(gradAfter[i]).toBe(0);
      }
      expect(maxErr).toBeLessThan(1e-5);
    };
    await expectAdam(gpuModel.weightsBuffers, gW, before.w);
    await expectAdam(gpuModel.latentsBuffers, gL, before.l);

    texture.dispose();
    gpuModel.dispose();
  });
});

describe('NTCTrainer', () => {
  const smallOptions = {
    gridChannels: 4,
    levels: 2,
    baseResolution: 16,
    hiddenSizes: [16],
    outputChannels: 4,
    batchSize: 1024,
    iterations: 40,
    seed: 1,
  };

  async function gradientTexture() {
    const rt = await bakeColorNodeToTexture(renderer, vec4(uv().x, uv().y, 0.5, 1), 64, {
      generateMipmaps: true,
    });
    return rt.texture;
  }

  it('is deterministic for a fixed seed and its loss decreases', async () => {
    const texture = await gradientTexture();
    const run = async () => {
      const losses: number[] = [];
      const result = await new NTCTrainer(smallOptions).train({
        renderer,
        sourceTexture: texture,
        onProgress: ({ loss }) => losses.push(loss),
      });
      return { result, losses };
    };
    const a = await run();
    const b = await run();
    expect(a.losses.length).toBeGreaterThan(2);
    expect(a.losses[a.losses.length - 1]).toBeLessThan(a.losses[0] * 0.5);
    expect(a.losses).toEqual(b.losses);
    for (let g = 0; g < a.result.cpuModel.grids.length; g++) {
      expect(a.result.cpuModel.grids[g].data).toEqual(b.result.cpuModel.grids[g].data);
    }
    for (let l = 0; l < a.result.cpuModel.decoder.layers.length; l++) {
      expect(a.result.cpuModel.decoder.layers[l].weights).toEqual(
        b.result.cpuModel.decoder.layers[l].weights,
      );
    }
    texture.dispose();
  });

  it('positionalEncoding + dualGrid training also converges', async () => {
    const texture = await gradientTexture();
    const losses: number[] = [];
    await new NTCTrainer({ ...smallOptions, positionalEncoding: true, dualGrid: true }).train({
      renderer,
      sourceTexture: texture,
      onProgress: ({ loss }) => losses.push(loss),
    });
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5);
    texture.dispose();
  });

  it('quantization-aware training leaves the exported latents exactly on uint4 levels', async () => {
    const texture = await gradientTexture();
    const result = await new NTCTrainer({ ...smallOptions, quantization: { mode: 'uint4' } }).train(
      { renderer, sourceTexture: texture },
    );
    expect(result.quantization.mode).toBe('uint4');
    expect(result.iterations).toBe(42); // 40 + 5% retrain
    const quantize = QUANTIZATION_SCHEMES.uint4.quantizeForwardCPU;
    result.cpuModel.grids.forEach((grid: any, g: number) => {
      const [lo, hi] = result.quantizationRange![g];
      expect(hi).toBeGreaterThan(lo);
      for (const value of grid.data)
        expect(Math.abs(value - quantize(value, lo, hi))).toBeLessThan(1e-6);
    });
    texture.dispose();
  });
});
