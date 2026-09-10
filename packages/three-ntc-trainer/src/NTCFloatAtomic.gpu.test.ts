import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { Fn, float, vec4 } from 'three/tsl';
import { getRenderer } from '../../../test/gpu-helpers.js';
import { createAdamParameterBuffers, disposeAdamParameterBuffers } from './NTCGPUKernelsTSL.js';
import { atomicAddFloat } from './NTCFloatAtomic.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { createTextureTrainBatchComputeNode, createTextureAdamWeightsComputeNode } from './NTCGPUComputeTSL.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';

for (const [count, contribution] of [[2048,1e-7], [524288,-0.0625]]) {
 it(`accumulates ${count} contributions of ${contribution} without integer overflow or truncation`, async () => {
  const r=await getRenderer(); const buffers=createAdamParameterBuffers(1);
  try {
   r.compute(Fn(()=>{atomicAddFloat(buffers.gradAtomic.element(0),float(contribution));})().compute(count));
   const value=new Float32Array(await r.getArrayBufferAsync(buffers.gradAttribute))[0];
   const expected=count*contribution;
   expect(Math.abs(value-expected)).toBeLessThan(Math.abs(expected)*1e-4);
   await commands.recordMetric({kind:'gradient-precision',count,contribution,expected,actual:value});
  } finally { disposeAdamParameterBuffers(buffers,r); }
 });
}

it('a nonzero sigmoid residual produces a corrective update below the old fixed-point threshold', async () => {
 const r=await getRenderer(); const metrics=[];
 const source=await bakeColorNodeToTexture(r,vec4(0),4);
 try {
  for (const gradientPrecision of ['fixed','float'] as const) {
   const options={gridChannels:1,levels:1,baseResolution:2,hiddenSizes:[],outputChannels:1,
    batchSize:4096,channelActivations:['sigmoid'],gradientPrecision};
   const cpu=createNTCGridPyramidModel(options,createRandom(1));
   cpu.grids[0].data.fill(0); cpu.decoder.layers[0].weights.fill(0);
   cpu.decoder.layers[0].biases.fill(Math.log(0.001/0.999));
   const gpu=new NTCGPUModel(options); gpu.initFromCPUModel(cpu);
   try {
    const train=createTextureTrainBatchComputeNode(gpu,[source.texture],{lod:float(0)});
    const adam=createTextureAdamWeightsComputeNode(gpu);
    for(let step=1;step<=40;step++) {
     gpu.stepUniform.value=step;
     r.compute(train); r.compute(adam);
    }
    const weights=new Float32Array(await r.getArrayBufferAsync(gpu.weightsBuffers.attribute));
    const bias=weights[gpu.layout.mlpLayers[0].biasesOffset];
    const mse=(1/(1+Math.exp(-bias)))**2;
    metrics.push({gradientPrecision,mse});
   } finally { gpu.dispose(); }
  }
  expect(metrics[1].mse).toBeLessThan(metrics[0].mse*0.65);
  await commands.recordMetric({kind:'sigmoid-dead-zone',measurements:metrics});
 } finally { source.dispose(); }
});
