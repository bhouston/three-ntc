import { floor, int } from 'three/tsl';

/** Extra mips covered by G0 before advancing through the feature pyramid.
 * Table 1: texture 1024 / grid 256 gives offset 2, hence the first band is
 * [0,1,2,3], followed by [4,5], [6,7], and the final open-ended band.
 */
export function computeFeatureLodOffset(textureResolution: number, baseResolution: number): number {
  return Math.max(0, Math.floor(Math.log2(Math.max(1,textureResolution)/Math.max(1,baseResolution))));
}

export function selectFeatureLevel(lod: number, levels: number, mipsPerLevel: number, lodOffset = 0): number {
  return Math.min(levels-1, Math.max(0, Math.floor((lod-lodOffset)/mipsPerLevel)));
}

export function selectFeatureLevelTSL(lod: any, levels: number, mipsPerLevel: number, lodOffset = 0): any {
  return int(floor(lod.sub(lodOffset).div(mipsPerLevel))).clamp(0,levels-1);
}
