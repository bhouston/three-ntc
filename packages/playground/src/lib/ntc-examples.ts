import { NTCLoader, NTCNodeMaterial, type MLPLayer } from 'three-ntc';

import { loadedModelInfo, type ModelInfoData } from './model-info.js';
import { getSharedRenderer } from '@/lib/renderer';

export const EXAMPLE_FILES = [
  { value: `${import.meta.env.BASE_URL}ntc/brick.ntc`, label: 'Brick' },
  { value: `${import.meta.env.BASE_URL}ntc/glass_dispersion.ntc`, label: 'Glass Dispersion' },
  { value: `${import.meta.env.BASE_URL}ntc/gold.ntc`, label: 'Gold' },
  { value: `${import.meta.env.BASE_URL}ntc/pearl.ntc`, label: 'Pearl' },
  { value: `${import.meta.env.BASE_URL}ntc/rainbow_emissive.ntc`, label: 'Rainbow Emissive' },
  { value: `${import.meta.env.BASE_URL}ntc/velvet.ntc`, label: 'Velvet' },
  { value: `${import.meta.env.BASE_URL}ntc/wave_normal.ntc`, label: 'Wave Normal' },
  { value: `${import.meta.env.BASE_URL}ntc/wood_flooring.ntc`, label: 'Wood Flooring' },
];

// Reconstructing one mip level finer than the raw screen-space estimate
// looks better in practice (this addon's tail-mip reconstruction tolerates
// the extra sharpness - see NTCNodeMaterial's computeAutoLodNode doc
// comment) - shared default for every place that builds/previews a
// material without its own explicit lodBias control.
export const DEFAULT_LOD_BIAS = 1;

export type LoadedMaterialGrid = {
  width: number;
  height: number;
  channels: number;
  bits: number;
}

export type LoadedMaterialMLPLayer = {
  inputSize: number;
  outputSize: number;
}

export type LoadedMaterial = {
  info: ModelInfoData;
  name: string;
  channels: string[];
  activeChannels: string[];
  constantChannels: string[];
  grids: LoadedMaterialGrid[];
  mlpLayers: LoadedMaterialMLPLayer[];
  material: NTCNodeMaterial;
}

export async function parseNtc(text: string): Promise<LoadedMaterial> {
  const manifest = JSON.parse(text);
  const loader = new NTCLoader();
  const { name, cpuModel, channelClassification } = loader.parse(manifest);
  const material = new NTCNodeMaterial(cpuModel, channelClassification, {
    renderer: await getSharedRenderer(),
    lodBias: DEFAULT_LOD_BIAS,
  });
  const info = loadedModelInfo(manifest, cpuModel, channelClassification, name ?? 'Untitled');
  const { activeChannels, constantChannels, grids } = info;
  const mlpLayers = (cpuModel?.decoder?.layers ?? []).map((l: MLPLayer) => ({
    inputSize: l.inputSize,
    outputSize: l.outputSize,
  }));
  return {
    info,
    name: name ?? 'Untitled',
    channels: [...activeChannels, ...constantChannels],
    activeChannels,
    constantChannels,
    grids,
    mlpLayers,
    material,
  };
}

export async function loadNtcFromUrl(url: string): Promise<LoadedMaterial> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  return parseNtc(await res.text());
}
