import { float, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { getRenderer } from '../../../test/gpu-helpers.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { trainingRandom } from './NTCSampling.js';
import { createTextureTrainBatchComputeNode, createTextureAdamLatentsComputeNode } from './NTCGPUComputeTSL.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';

it('noise is applied to stored taps, matches the CPU counter stream, and switches off for frozen adaptation', async () => {
  const renderer=await getRenderer();
  const options={gridChannels:1,levels:1,baseResolution:32,textureResolution:32,
    hiddenSizes:[],batchSize:2048,outputChannels:4,quantization:{mode:'uint4'}};
  const cpu=createNTCGridPyramidModel(options,createRandom(1));cpu.grids[0].data.fill(0);
  const gpu=new NTCGPUModel(options);gpu.initFromCPUModel(cpu);
  const source=await bakeColorNodeToTexture(renderer,vec4(0),32);
  const batch=createTextureTrainBatchComputeNode(gpu,[source.texture],{lod:float(0)});
  renderer.compute(batch);
  const values=new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute));
  let mean=0,variance=0;
  for(let i=0;i<2048;i++) {
    const x=Math.floor(trainingRandom(i,1,0)*32),y=Math.floor(trainingRandom(i,1,1)*32);
    const expected=(trainingRandom(y*32+x,1,11)-0.5)/16;
    const actual=values[i*gpu.layout.activationStride];
    expect(actual).toBeCloseTo(expected,7);
    mean+=actual/2048;variance+=actual*actual/2048;
  }
  expect(Math.abs(mean)).toBeLessThan(0.002);
  expect(Math.abs(variance-1/(16*16*12))).toBeLessThan(0.00005);
  gpu.quantizationNoiseUniform.value=0;renderer.compute(batch);
  const frozen=new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute));
  for(let i=0;i<2048;i++) expect(frozen[i*gpu.layout.activationStride]).toBe(0);
  // Force a huge outward Adam update; the stored parameters must remain bounded.
  gpu.learningRateUniform.value=10;
  gpu.latentsBuffers.gradAttribute.array.fill(-100000);
  gpu.latentsBuffers.gradAttribute.needsUpdate=true;
  renderer.compute(createTextureAdamLatentsComputeNode(gpu));
  const clamped=new Float32Array(await renderer.getArrayBufferAsync(gpu.latentsBuffers.attribute));
  expect([...clamped].every(x=>x===0.4375)).toBe(true);
  source.dispose();gpu.dispose();
});

it('a fixed degenerate quantization range stays finite on the GPU', async () => {
  const renderer=await getRenderer();
  const options={gridChannels:1,levels:1,baseResolution:2,hiddenSizes:[],batchSize:1,
    outputChannels:4,quantization:{mode:'uint4',range:[0.25,0.25] as [number,number]}};
  const gpu=new NTCGPUModel(options);gpu.initFromCPUModel(createNTCGridPyramidModel(options,createRandom(1)));
  const source=await bakeColorNodeToTexture(renderer,vec4(0),2);
  renderer.compute(createTextureTrainBatchComputeNode(gpu,[source.texture],{lod:float(0)}));
  const values=new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute));
  expect(values[0]).toBe(0.25);
  source.dispose();gpu.dispose();
});
