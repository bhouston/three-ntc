import { expect, it, vi } from 'vitest';
import { commands } from 'vitest/browser';
import { buildChannelActivations } from 'three-ntc';
import { getRenderer } from '../../../test/gpu-helpers.js';
import { MaterialXLoader, classifyMaterialChannels, bakeMaterialToTextures, NTCTrainer } from './index.js';
import brick from '../../website/public/materialx/brick.mtlx?raw';

it('trains the default brick with identical results after an atomic compiler failure', async () => {
  const renderer = await getRenderer();
  const asset: any = new MaterialXLoader().parseBuffer(new TextEncoder().encode(brick).buffer,
    'brick.mtlx', { uvSpace: 'top-left', throwOnErrors: true });
  if (asset.texturesReady) await asset.texturesReady;
  const material: any = Object.values(asset.materials ?? asset)[0];
  const classification = classifyMaterialChannels(material);
  const targets = await bakeMaterialToTextures(renderer, material, 1024, classification.activeChannels);
  const device = renderer.backend.device;
  const createPipeline = device.createComputePipelineAsync.bind(device);
  const compiler = vi.spyOn(device, 'createComputePipelineAsync').mockImplementation((descriptor: any) => {
    if (descriptor.label === 'NTC float accumulation capability check') {
      return Promise.reject(new Error('field may not be qualified with an address space'));
    }
    return createPipeline(descriptor);
  });
  const metrics: any[] = [];
  const weights: number[][] = [];
  try {
    for (const gradientPrecision of ['fixed', 'auto'] as const) {
      const trainer = new NTCTrainer({ gradientPrecision, gridChannels: 4, levels: 4,
        baseResolution: 256, hiddenSizes: [32,32], hiddenActivation: 'relu',
        positionalEncoding: true, dualGrid: true, outputChannels: classification.totalChannels,
        channelActivations: buildChannelActivations(classification.activeChannels),
        batchSize: 8192, iterations: 10000, learningRate: 0.01, seed: 1,
        quantization: { mode: 'uint8' } as any });
      const losses: number[] = [];
      const start = performance.now();
      const result = await trainer.train({ renderer, sourceTextures: targets.map(t => t.texture),
        onProgress: p => {
          expect(p.gpuModel.floatGradients).toBe(false);
          expect(Number.isFinite(p.loss)).toBe(true);
          expect(p.loss).toBeGreaterThan(0);
          losses.push(p.loss);
          if (p.iteration >= 129) trainer.abort();
        } });
      weights.push(result.cpuModel.decoder.layers.flatMap(l => [...l.weights, ...l.biases]));
      expect(losses.at(-1)!).toBeLessThan(losses[0] * 0.2);
      metrics.push({ gradientPrecision, iterations: 129, firstLoss: losses[0], lastLoss: losses.at(-1),
        trainingMs: performance.now() - start });
    }
    expect(compiler).toHaveBeenCalled();
    expect(weights[1]).toEqual(weights[0]);
    expect(metrics[1].lastLoss).toBe(metrics[0].lastLoss);
    await commands.recordMetric({ kind: 'default-brick-safari-fallback', sourceResolution: 1024,
      batchSize: 8192, scheduleIterations: 10000, simulatedCompilerFailure: true, metrics });
  } finally {
    compiler.mockRestore();
    targets.forEach(t => t.dispose());
    material.dispose();
  }
}, 120000);
