import { expect, it } from 'vitest';
import { computeFeatureLodOffset, selectFeatureLevel } from './NTCMipBands.js';

it('reproduces Table 1 including the wide first band and final tail',()=>{
  const offset=computeFeatureLodOffset(1024,256);
  expect(offset).toBe(2);
  expect(Array.from({length:11},(_,lod)=>selectFeatureLevel(lod,4,2,offset)))
    .toEqual([0,0,0,0,1,1,2,2,3,3,3]);
  expect(selectFeatureLevel(3.99,4,2,offset)).toBe(0);
  expect(selectFeatureLevel(4,4,2,offset)).toBe(1);
  expect(selectFeatureLevel(100,4,2,offset)).toBe(3);
});

it('adjusts bands to the source/grid ratio and preserves legacy metadata',()=>{
  expect(computeFeatureLodOffset(2048,256)).toBe(3);
  expect(computeFeatureLodOffset(128,256)).toBe(0);
  expect(selectFeatureLevel(2,4,2)).toBe(1);
  expect(selectFeatureLevel(2,4,2,2)).toBe(0);
});
