import { float, floor, log, uint, vec2 } from 'three/tsl';

// Counter-based integer hashing: separate streams for U, V, mip, and mixture.
// Keeping the top 24 bits makes the float conversion exact on CPU and GPU.
export function trainingRandom(sample: number, step: number, stream: number): number {
  let x = (sample + Math.imul(step, 0x9e3779b9) + Math.imul(stream, 0x85ebca6b)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  return ((x ^ (x >>> 16)) >>> 8) / 16777216;
}

export function trainingRandomTSL(sample: any, step: any, stream: number): any {
  const a = uint(sample).add(uint(step).mul(uint(0x9e3779b9))).add(uint(stream).mul(uint(0x85ebca6b))).toVar();
  a.assign(a.bitXor(a.shiftRight(16)).mul(uint(0x7feb352d)));
  a.assign(a.bitXor(a.shiftRight(15)).mul(uint(0x846ca68b)));
  return float(a.bitXor(a.shiftRight(16)).shiftRight(8)).div(16777216);
}

export function trainingUVTSL(sample: any, step: any): any {
  return vec2(trainingRandomTSL(sample,step,0), trainingRandomTSL(sample,step,1));
}

export function trainingLodTSL(sample: any, step: any, maxLod: number): any {
  const area = floor(log(trainingRandomTSL(sample,step,2).max(1 / 16777216)).div(-Math.log(4)));
  const uniform = floor(trainingRandomTSL(sample,step,3).mul(maxLod+1));
  return trainingRandomTSL(sample,step,4).lessThan(0.05).select(uniform,area).clamp(0,maxLod);
}
