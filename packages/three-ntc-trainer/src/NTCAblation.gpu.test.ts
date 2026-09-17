import { float, vec2, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { NTCLoader } from 'three-ntc';
import { getRenderer, runtimeFeaturesCpu } from '../../../test/gpu-helpers.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createTextureTrainBatchComputeNode } from './NTCGPUComputeTSL.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { encodeNTC } from './NTCManifest.js';
import { NTCTrainer } from './NTCTrainer.js';
import { createRandom } from './NTCTrainingUtils.js';

for(const g0Sampling of ['bilinear','concatenate'] as const)for(const positionalEncoding of [false,true]) {
  it(`independent encoding=${positionalEncoding}, G0=${g0Sampling}: GPU input and export agree`,async()=>{
    const renderer=await getRenderer();
    const options={gridChannels:2,lowResChannels:1,levels:2,baseResolution:8,textureResolution:32,
      positionalEncoding,positionalEncodingPeriod:8,g0Sampling,dualGrid:true,hiddenSizes:[],batchSize:1,outputChannels:3};
    const cpu=createNTCGridPyramidModel(options,createRandom(7));
    const gpu=new NTCGPUModel(options);gpu.initFromCPUModel(cpu);
    const source=await bakeColorNodeToTexture(renderer,vec4(0),32);
    try {
      renderer.compute(createTextureTrainBatchComputeNode(gpu,[source.texture],{uv:vec2(0.3,0.4),lod:float(0)}));
      const data=new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute));
      const expected=runtimeFeaturesCpu(cpu,0.3,0.4,0);
      expect(gpu.layout.inputSize).toBe((g0Sampling==='concatenate'?8:2)+(positionalEncoding?12:0)+1+1);
      expected.forEach((value,i)=>expect(data[i]).toBeCloseTo(value,3));
      const loaded=new NTCLoader().parse(encodeNTC(cpu,{activeChannels:[{key:'albedo'}],constantValues:{}})).cpuModel;
      expect(loaded.g0Sampling).toBe(g0Sampling);
      expect(runtimeFeaturesCpu(loaded,0.3,0.4,0).length).toBe(gpu.layout.inputSize);
    } finally {source.dispose();gpu.dispose();}
  });
}

it('microbatches preserve a full logical batch and one optimizer update',async()=>{
  const renderer=await getRenderer();
  const source=await bakeColorNodeToTexture(renderer,vec4(0.3,0.5,0.7,1),16,{generateMipmaps:true});
  const models:any[]=[];const losses:number[]=[];
  try {
    for(const microBatchSize of [64,16]) {
      const result=await new NTCTrainer({gridChannels:2,levels:2,baseResolution:4,textureResolution:16,
        hiddenSizes:[4],batchSize:64,microBatchSize,iterations:3,seed:7,gradientPrecision:'float',outputChannels:3}).train({renderer,sourceTexture:source.texture});
      expect(result.iteration).toBe(3);models.push(result.cpuModel);losses.push(result.loss);
    }
    const flatten=(m:any)=>[...m.grids.flatMap((g:any)=>[...g.data]),...m.decoder.layers.flatMap((l:any)=>[...l.weights,...l.biases])];
    const a=flatten(models[0]),b=flatten(models[1]);a.forEach((v:number,i:number)=>expect(b[i]).toBeCloseTo(v,5));
    expect(losses[1]).toBeCloseTo(losses[0],5);
  } finally {source.dispose();}
});
