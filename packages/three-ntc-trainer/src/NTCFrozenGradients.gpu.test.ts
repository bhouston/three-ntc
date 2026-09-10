import { float, vec2, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { createTextureTrainBatchComputeNode, createAccumulateGradientNormComputeNode } from './NTCGPUComputeTSL.js';
import { createResetGradientsComputeNode, createResetGradientNormComputeNode } from './NTCGPUKernelsTSL.js';
import { getRenderer } from '../../../test/gpu-helpers.js';

it('frozen-grid steps neither accumulate latent gradients nor include them in clipping', async () => {
  const renderer=await getRenderer();
  const options={gridChannels:1,levels:1,baseResolution:2,hiddenSizes:[4],batchSize:1,outputChannels:4};
  const gpu=new NTCGPUModel(options);
  gpu.initFromCPUModel(createNTCGridPyramidModel(options,createRandom(1)));
  const source=await bakeColorNodeToTexture(renderer,vec4(0.5),4);
  const batch=createTextureTrainBatchComputeNode(gpu,[source.texture],
    {uv:vec2(0.45,0.5),lod:float(0),trainLatents:false});
  renderer.compute(createResetGradientsComputeNode(gpu));
  for(let i=0;i<5;i++) renderer.compute(batch);
  const grads=new Int32Array(await renderer.getArrayBufferAsync(gpu.latentsBuffers.gradAttribute));
  expect([...grads].every(x=>x===0)).toBe(true);
  gpu.latentsBuffers.gradAttribute.array.fill(100000000);
  gpu.latentsBuffers.gradAttribute.needsUpdate=true;
  gpu.weightsBuffers.gradAttribute.array.fill(0);
  gpu.weightsBuffers.gradAttribute.array[0]=100000;
  gpu.weightsBuffers.gradAttribute.needsUpdate=true;
  renderer.compute(createResetGradientNormComputeNode(gpu));
  renderer.compute(createAccumulateGradientNormComputeNode(gpu,false));
  const norm=new Int32Array(await renderer.getArrayBufferAsync(gpu.gradNormAttribute))[0]/100000;
  expect(norm).toBe(1);
  gpu.dispose();source.dispose();
});
