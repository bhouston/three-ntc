import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

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

export interface LoadedMaterial {
  name: string;
  channels: string[];
  material: any;
}

export async function parseNtc(text: string): Promise<LoadedMaterial> {
  const manifest = JSON.parse(text);
  const loader = new NTCLoader();
  const { name, cpuModel, channelClassification } = loader.parse(manifest);
  const material = new NTCNodeMaterial(cpuModel, channelClassification, { renderer: getSharedRenderer() });
  const channels = Object.keys(channelClassification ?? {}).filter((key) => Boolean((channelClassification as any)[key]));
  return { name: name ?? 'Untitled', channels, material };
}

export async function loadNtcFromUrl(url: string): Promise<LoadedMaterial> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  return parseNtc(await res.text());
}
