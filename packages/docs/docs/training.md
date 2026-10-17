---
title: Training and export
sidebar_position: 3
description: Fit a material in the browser and save a reusable neural texture asset.
---

Training is an offline preparation step performed with browser GPU compute. The exported asset needs only `three-ntc` at runtime. The interactive trainer on this site provides a UI for this workflow; use the library when integrating it into your own authoring tools.

```sh
npm install three three-ntc three-ntc-trainer
```

Use an initialized three.js `WebGPURenderer` on a hardware GPU. Serve over HTTPS or localhost. Keep the trainer in client-only code and surface initialization, shader compilation, and device-limit errors in your UI. Large profiles can require substantial compute time and memory.

## Fit a node material

`fitNTCMaterial()` classifies the source's channels, bakes spatially varying channels into textures, trains a shared model, and returns a result containing the trained `material`, `cpuModel`, channel classification, and final metrics.

This example creates a procedural source without any external images. `renderer` is your initialized WebGPU renderer and `previewMesh` is an existing mesh in your scene:

```ts
import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import { mix, sin, uv, vec3 } from 'three/tsl';
import { fitNTCMaterial, NTCExporter, type NTCTrainProgress } from 'three-ntc-trainer';
import type { NTCNodeMaterial } from 'three-ntc';

const source = new MeshPhysicalNodeMaterial();
const stripe = sin(uv().x.mul(40)).mul(0.5).add(0.5);
source.colorNode = mix(vec3(0.04, 0.12, 0.3), vec3(0.8, 0.5, 0.1), stripe);
source.roughness = 0.35;
source.metalness = 0.2;

try {
  const result = await fitNTCMaterial(renderer, source, {
    resolution: 256,
    baseResolution: 64,
    levels: 3,
    hiddenSizes: [32, 32],
    iterations: 3000,
    quantization: { mode: 'uint8' },
    onProgress: ({ iteration, iterations, loss, material }: NTCTrainProgress & { material: NTCNodeMaterial }) => {
      previewMesh.material = material;
      console.log(`${iteration}/${iterations}: ${loss}`);
    },
  });
  previewMesh.material = result.material;

  const manifest = new NTCExporter().parse(result.material, {
    name: 'Procedural stripes',
    source: 'Procedural TSL example',
  });
  const blob = new Blob([JSON.stringify(manifest)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'stripes.ntc';
  link.click();
  // Delay revocation until the browser has started consuming the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);

  // Keep result.material alive while previewMesh uses it.
  // Detach it and call result.material.dispose() when the preview is removed.
} finally {
  source.dispose();
}
```

Progress uses `iteration`, not `step`. Each callback's material is a temporary preview: the helper disposes its predecessor when rebuilding. Replace the mesh's material immediately and do not retain old previews for later use. The helper disposes its baked render targets after training; your application owns the returned material, source resources, and renderer.

A material whose channels are all constant causes `fitNTCMaterial()` to throw because there is nothing spatially varying to train. Keep such a source as an ordinary physical material.

## MaterialX sources

The trainer package exports `MaterialXLoader` and channel-baking helpers. The site's trainer includes example MaterialX materials and handles the associated loading and preview flow. MaterialX image dependencies must be available at the paths expected by the document, with appropriate CORS headers when hosted on another origin.

Only graphs and surface mappings supported by this implementation can be baked faithfully. Inspect the original preview before training. Source-loading or graph-translation mistakes cannot be corrected by more training. For transformed image graphs, the exported `inferAlbedoUvTransform` helper can identify a local UV transform; pass a consistent `uvTransform` into fitting so repeated content is baked once and queried correctly.

## Choose a budget

Start with a small bake resolution and model, confirm the source channels, then increase capacity or iterations after inspecting the result.

| Option                                     | Effect                                                            |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `resolution`                               | Bake resolution used by `fitNTCMaterial()`; defaults to 512.      |
| `baseResolution`, `levels`, `gridChannels` | Feature-grid capacity, memory, and exported size.                 |
| `hiddenSizes`, `hiddenActivation`          | Decoder capacity and per-sample shading cost.                     |
| `iterations`, `batchSize`                  | Optimization budget and GPU workload. Defaults are 3000 and 4096. |
| `learningRate`, `weightsLearningRate`      | Optimization rates for latents and optionally decoder weights.    |
| `dualGrid`, `positionalEncoding`           | Optional richer decoder input with increased cost.                |
| `quantization`                             | Latent storage precision and quantization-aware training.         |

`getNTCProfile(name)` returns model-shape options for `mobile-fast`, `mobile-balanced`, or `desktop-quality`; spread them into the fit options. Names describe intended cost tiers, not a device compatibility guarantee or fixed bitrate. Profile values can differ from the bare trainer defaults.

`getNTCPaperProfile(textureResolution)` opts into a much larger paper-oriented configuration: 250,000 iterations with batches of 524,288 samples. It requires a power-of-two source resolution of at least 16. This is an experimental research budget, substantially more expensive than the browser defaults; it does not promise the paper's measured quality or bitrate.

## Quantization

The trainer defaults to no quantization-aware training. Choose `uint8`, `uint4`, or `uint2` to simulate latent rounding or noise during optimization and write the corresponding latent codec. The default method is noise, with automatic ranges per level. A final fraction of training (default 5%) freezes hard-quantized latents while updating the decoder to absorb rounding error.

Only latent quantization is implemented; selecting weight quantization is rejected. Decoder weights are serialized with float16 packing. Compare the **reloaded exported asset** with the source to assess actual shipped quality.

## Direct training and cancellation

For a pre-baked texture or custom channel packing, construct `NTCTrainer` directly:

```ts
import { NTCTrainer } from 'three-ntc-trainer';

const trainer = new NTCTrainer({ outputChannels: 3, iterations: 3000 });
// Connect your cancel button to trainer.abort().
const result = await trainer.train({ renderer, sourceTexture });
console.log(result.iteration, result.loss, result.stoppedEarly);
```

`sourceTexture` must contain the intended channel data and mip chain. Direct callers are responsible for matching `outputChannels`, activation layout, channel classification, and the eventual runtime material. The trainer frees its temporary GPU model; it does not take ownership of caller-supplied source textures.

`abort()` requests completion after the current iteration and returns the current model with `stoppedEarly: true` when it stops before the requested budget. The convenience `fitNTCMaterial()` helper does not expose its internal trainer for cancellation; use a direct trainer for a cancellable application workflow.

The default `gradientPrecision: 'auto'` probes floating-point accumulation and falls back to fixed-point when the GPU shader cannot compile it. Explicit `'float'` throws if unsupported. Compare convergence on the hardware you intend to support.
