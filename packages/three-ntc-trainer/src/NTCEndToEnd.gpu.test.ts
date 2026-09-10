// End-to-end browser/WebGPU tests: the full pipeline a user of this repo
// runs - classify + bake a material, train, build an NTCNodeMaterial, export
// a .ntc with NTCExporter, reload it with NTCLoader - judged against the
// baked source by PSNR, and the reloaded model against the in-memory one.
// Runs under the `gpu` vitest project (see /vitest.config.ts).
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { float, sin, uv, vec3 } from 'three/tsl';
import { NTCLoader, NTCNodeMaterial } from 'three-ntc';
import { beforeAll, describe, expect, it } from 'vitest';

import { fitNTCMaterial } from './NTCFit.js';
import { NTCExporter } from './NTCExporter.js';
import { bakeMaterialToTextures, classifyMaterialChannels } from './NTCSource.js';
import { MaterialXLoader } from './materialx/MaterialXLoader.js';
import checkerboardMtlx from '../../website/public/materialx/checkerboard.mtlx?raw';
import {
  getRenderer,
  maxAbsDiff,
  psnr,
  readRenderTargetFloats,
  renderNodeToFloats,
} from '../../../test/gpu-helpers.js';

const SIZE = 64;

let renderer: any;
beforeAll(async () => {
  renderer = await getRenderer();
});

const trainerOptions = {
  resolution: SIZE,
  levels: 2,
  baseResolution: 32,
  hiddenSizes: [16, 16],
  batchSize: 4096,
  iterations: 400,
  seed: 1,
  quantization: { mode: 'uint8' },
};

/** Albedo as the pipeline sees it: the first bake target's RGB. */
async function bakeAlbedo(material: any): Promise<Float32Array> {
  const classification = classifyMaterialChannels(material);
  expect(classification.activeChannels[0].key).toBe('albedo');
  const [rt] = await bakeMaterialToTextures(
    renderer,
    material,
    SIZE,
    classification.activeChannels,
  );
  const pixels = await readRenderTargetFloats(renderer, rt, SIZE);
  rt.dispose();
  return pixels;
}

function renderAlbedo(cpuModel: any, classification: any): Promise<Float32Array> {
  const material = new NTCNodeMaterial(cpuModel, classification, {
    debugView: 'albedo',
    lodNode: float(0),
  });
  const pixels = renderNodeToFloats(renderer, (material as any).colorNode, SIZE);
  pixels.finally(() => material.dispose());
  return pixels;
}

/** fit -> render -> export -> reload -> render, returning what to assert on. */
async function fitExportReload(material: any) {
  const source = await bakeAlbedo(material);
  const losses: number[] = [];
  const fit = await fitNTCMaterial(renderer, material, {
    ...trainerOptions,
    onProgress: ({ loss }: any) => losses.push(loss),
  });
  const trained = await renderAlbedo(fit.cpuModel, fit.channelClassification);

  const manifest = new NTCExporter().parse(fit.material, { name: 'test' });
  const reloaded = new NTCLoader().parse(JSON.stringify(manifest));
  const roundTripped = await renderAlbedo(reloaded.cpuModel, reloaded.channelClassification);

  fit.material.dispose();
  return { losses, source, trained, roundTripped, manifest, reloaded, fit };
}

describe('fit -> export -> load -> render', () => {
  it('reconstructs a smooth procedural albedo and survives the .ntc round trip', async () => {
    const material = new MeshPhysicalNodeMaterial();
    const u = uv();
    (material as any).colorNode = vec3(
      u.x,
      u.y,
      sin(u.x.mul(6.283))
        .mul(sin(u.y.mul(6.283)))
        .mul(0.25)
        .add(0.5),
    );

    const { losses, source, trained, roundTripped, manifest, reloaded } =
      await fitExportReload(material);

    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.2);
    expect(psnr(source, trained)).toBeGreaterThan(30);

    // Export metadata carries the training configuration through.
    expect(manifest.latents.levels.map((l: any) => l.dtype)).toEqual(['uint8', 'uint8']);
    expect(reloaded.channelClassification.activeChannels.map((c: any) => c.key)).toEqual([
      'albedo',
    ]);
    expect(reloaded.cpuModel.maxLod).toBe(Math.log2(SIZE));

    // QAT trained against uint8 rounding, and the MLP is stored as float16:
    // the reloaded model must decode (almost) exactly like the in-memory one.
    expect(maxAbsDiff(trained, roundTripped)).toBeLessThan(0.02);
    expect(psnr(trained, roundTripped)).toBeGreaterThan(45);
    expect(psnr(source, roundTripped)).toBeGreaterThan(30);
    material.dispose();
  });

  it('fits a MaterialX material end to end', async () => {
    const { materials } = new MaterialXLoader().parse(checkerboardMtlx, {
      uvSpace: 'top-left',
      throwOnErrors: true,
    });
    const material = Object.values(materials)[0] as any;
    expect(material).toBeDefined();

    const { losses, source, trained, roundTripped, fit } = await fitExportReload(material);

    // MaterialX wires roughness/specular nodes too; albedo is packed first.
    expect(fit.channelClassification.activeChannels[0].key).toBe('albedo');
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5);
    // A 4x4 hard-edged checkerboard on a 32-texel grid: coarse but clearly learned.
    expect(psnr(source, trained)).toBeGreaterThan(20); // measured ~27.5 dB
    expect(psnr(trained, roundTripped)).toBeGreaterThan(40);
    material.dispose();
  });
});
