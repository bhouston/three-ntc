import { constantEqualsDefault, getChannel, NTCLoader, NTCNodeMaterial } from 'three-ntc';

import { getSharedRenderer } from '@/lib/renderer';

export const EXAMPLE_FILES = [
  { value: '/ntc/brick.ntc', label: 'Brick' },
  { value: '/ntc/glass_dispersion.ntc', label: 'Glass Dispersion' },
  { value: '/ntc/gold.ntc', label: 'Gold' },
  { value: '/ntc/pearl.ntc', label: 'Pearl' },
  { value: '/ntc/rainbow_emissive.ntc', label: 'Rainbow Emissive' },
  { value: '/ntc/velvet.ntc', label: 'Velvet' },
  { value: '/ntc/wave_normal.ntc', label: 'Wave Normal' },
  { value: '/ntc/wood_flooring.ntc', label: 'Wood Flooring' },
];

// Reconstructing one mip level finer than the raw screen-space estimate
// looks better in practice (this addon's tail-mip reconstruction tolerates
// the extra sharpness - see NTCNodeMaterial's computeAutoLodNode doc
// comment) - shared default for every place that builds/previews a
// material without its own explicit lodBias control.
export const DEFAULT_LOD_BIAS = 1;

export interface LoadedMaterialGrid {
  width: number;
  height: number;
  channels: number;
  bits: number;
}

export interface LoadedMaterialMLPLayer {
  inputSize: number;
  outputSize: number;
}

export interface LoadedMaterial {
  name: string;
  channels: string[];
  activeChannels: string[];
  constantChannels: string[];
  grids: LoadedMaterialGrid[];
  mlpLayers: LoadedMaterialMLPLayer[];
  material: any;
}

export async function parseNtc(text: string): Promise<LoadedMaterial> {
  const manifest = JSON.parse(text);
  const loader = new NTCLoader();
  const { name, cpuModel, channelClassification } = loader.parse(manifest);
  const material = new NTCNodeMaterial(cpuModel, channelClassification, {
    renderer: await getSharedRenderer(),
    lodBias: DEFAULT_LOD_BIAS,
  });
  const activeChannels = (channelClassification?.activeChannels ?? []).map((c: any) => c.key);
  // A constant channel that just equals its declared default (i.e. was never
  // meaningfully set on the source material) isn't actionable info for a
  // viewer - drop it the same way NTCNodeMaterial itself skips applying it
  // (see NTCFormat's constantEqualsDefault).
  const constantChannels = Object.entries(channelClassification?.constantValues ?? {})
    .filter(([key, value]) => !constantEqualsDefault(getChannel(key), value))
    .map(([key]) => key);
  // Every stored grid level is quantized to a fixed bit depth per the
  // `.ntc` format (`NTCManifestLevel.dtype`, currently always 'uint8') -
  // read straight off the manifest since `cpuModel.grids` (already decoded
  // to float) no longer carries it.
  const grids = (cpuModel?.grids ?? []).map((g: any, i: number) => ({
    width: g.width,
    height: g.height,
    channels: g.channels,
    bits: parseInt((manifest?.latents?.levels?.[i]?.dtype ?? 'uint8').replace(/\D/g, ''), 10),
  }));
  const mlpLayers = (cpuModel?.decoder?.layers ?? []).map((l: any) => ({ inputSize: l.inputSize, outputSize: l.outputSize }));
  return {
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
