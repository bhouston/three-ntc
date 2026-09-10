import { float, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { getRenderer } from '../../../test/gpu-helpers.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { trainingRandom } from './NTCSampling.js';
import { createTextureTrainBatchComputeNode } from './NTCGPUComputeTSL.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';

it('training visits exact texel centers instead of continuous filtered coordinates', async () => {
  const renderer=await getRenderer();
  const options={gridChannels:1,levels:1,baseResolution:4,textureResolution:4,
    hiddenSizes:[],batchSize:128,outputChannels:4};
  const cpu=createNTCGridPyramidModel(options,createRandom(1));
  cpu.grids[0].data.set(Array.from({length:16},(_,i)=>i));
  const gpu=new NTCGPUModel(options); gpu.initFromCPUModel(cpu);
  const source=await bakeColorNodeToTexture(renderer,vec4(0),4);
  renderer.compute(createTextureTrainBatchComputeNode(gpu,[source.texture],{lod:float(0)}));
  const values=new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute));
  for(let i=0;i<128;i++) {
    const x=Math.floor(trainingRandom(i,gpu.stepUniform.value,0)*4);
    const y=Math.floor(trainingRandom(i,gpu.stepUniform.value,1)*4);
    expect(values[i*gpu.layout.activationStride]).toBe(y*4+x);
  }
  source.dispose();gpu.dispose();
});
