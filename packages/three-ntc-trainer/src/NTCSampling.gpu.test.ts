import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, float, instanceIndex, int, storage } from 'three/tsl';
import { expect, it } from 'vitest';
import { trainingRandom, trainingRandomTSL } from './NTCSampling.js';
import { getRenderer } from '../../../test/gpu-helpers.js';

it('integer sample streams agree exactly on CPU and GPU', async () => {
  const renderer = await getRenderer();
  const attr = new StorageBufferAttribute(new Float32Array(1024*5),1,Float32Array);
  const buffer = storage(attr,'float',1024*5);
  renderer.compute(Fn(()=>{
    const s=int(instanceIndex);
    for(let stream=0;stream<5;stream++) buffer.element(s.mul(5).add(stream))
      .assign(trainingRandomTSL(s,float(10000),stream));
  })().compute(1024));
  const data=new Float32Array(await renderer.getArrayBufferAsync(attr));
  for(let s=0;s<1024;s++) for(let stream=0;stream<5;stream++)
    expect(data[s*5+stream]).toBe(trainingRandom(s,10000,stream));
  attr.dispose();
});
