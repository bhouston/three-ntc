import { expect, it } from 'vitest';
import { trainingRandom } from './NTCSampling.js';

it('covers the entire UV domain independently of mip for non-square batches', () => {
  const bins = Array.from({length:4},()=>({n:0,u:0,v:0, left:0,right:0,top:0}));
  for(let step=1;step<=32;step++) for(let s=0;s<8192;s++) {
    const u=trainingRandom(s,step,0), v=trainingRandom(s,step,1);
    const lod=Math.floor(-Math.log(Math.max(1/16777216,trainingRandom(s,step,2)))/Math.log(4));
    expect(u>=0 && u<1 && v>=0 && v<1).toBe(true);
    if(lod>3) continue;
    const b=bins[lod]; b.n++; b.u+=u; b.v+=v;
    if(u<0.1)b.left++; if(u>0.9)b.right++; if(v>90/91)b.top++;
  }
  for(const b of bins) {
    expect(Math.abs(b.u/b.n-0.5)).toBeLessThan(0.025);
    expect(Math.abs(b.v/b.n-0.5)).toBeLessThan(0.025);
    expect(b.left/b.n).toBeGreaterThan(0.07);
    expect(b.right/b.n).toBeGreaterThan(0.07);
    expect(b.top).toBeGreaterThan(0);
  }
});
