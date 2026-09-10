import { float, vec2, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { getRenderer } from '../../../test/gpu-helpers.js';
import { NTCGPUModel } from './NTCGPUModel.js';
import { createNTCGridPyramidModel } from './NTCGridPyramidModel.js';
import { createRandom } from './NTCTrainingUtils.js';
import { createTextureTrainBatchComputeNode } from './NTCGPUComputeTSL.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';

it('PE repeats every eight target texels rather than every G0 cell', async () => {
  const renderer=await getRenderer();
  const options={gridChannels:1,levels:1,baseResolution:8,textureResolution:32,
    positionalEncoding:true,hiddenSizes:[],batchSize:1,outputChannels:4};
  const gpu=new NTCGPUModel(options);gpu.initFromCPUModel(createNTCGridPyramidModel(options,createRandom(1)));
  const source=await bakeColorNodeToTexture(renderer,vec4(0),32,{generateMipmaps:true});
  for(const lod of [0,1]) {
    const read=async (x:number) => {
      renderer.compute(createTextureTrainBatchComputeNode(gpu,[source.texture],
        {uv:vec2((x+0.5)/(32>>lod),0.5/(32>>lod)),lod:float(lod)}));
      return [...new Float32Array(await renderer.getArrayBufferAsync(gpu.activationsAttribute)).slice(4,16)];
    };
    const a=await read(0),b=await read(8),c=await read(4);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  }
  source.dispose();gpu.dispose();
});
