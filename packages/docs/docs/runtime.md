---
title: Runtime integration
sidebar_position: 2
description: Load assets, choose reconstruction filtering, and manage material resources.
---

## Loading and ownership

`NTCLoader` extends the three.js `Loader` API. `loadAsync(url)` fetches and validates the JSON manifest; `parse()` accepts an already parsed manifest or a JSON string. Both return `{ name, cpuModel, channelClassification }`. Loading reconstructs the CPU model; creating and rendering `NTCNodeMaterial` uploads the model's GPU resources.

```ts
import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

const loader = new NTCLoader();
loader.setPath('/materials/');
const asset = await loader.loadAsync('gold.ntc');
const material = new NTCNodeMaterial(
  asset.cpuModel,
  asset.channelClassification,
  { samplingMode: 'nearest', lodBias: 0 },
);

// Attach to a mesh rendered by three.js WebGPURenderer.
mesh.material = material;

// When no mesh uses it anymore:
material.dispose();
```

Catch load errors at your application's boundary and present a useful failure state. Cross-origin asset hosts must allow your origin with CORS headers. The `.ntc` extension contains JSON, so it can be served as `application/json` on ordinary static hosting.

`NTCNodeMaterial` extends `MeshPhysicalNodeMaterial`. It reconstructs spatially varying channels with the shared decoder and applies constant channels directly. Use lighting, environment maps, geometry normals, and UVs as you would for other physical node materials. Dispose geometry, controls, environment textures, and the renderer separately; material disposal releases its own feature textures.

## Reconstruction filtering

Filtering is a runtime choice independent of training and export:

| Mode | Decoder evaluations per sample | Result |
| --- | --- | --- |
| `nearest` (default) | 1 | Nearest physical texel at the nearest mip; lowest decode cost. |
| `stochastic` | 1 | Random texel and mip sampled with trilinear probabilities; introduces noise. |
| `trilinear` | 8 | Four reconstructed texels at each of two mips, blended deterministically. |

```ts
material.setSamplingMode('trilinear');
material.samplingMode = 'nearest'; // Equivalent property setter.
material.setLodBias(1);            // Positive selects finer mips.
```

Changing sampling mode rebuilds the TSL graph and triggers shader compilation on the next render. Selecting the existing mode is a no-op. LOD bias changes a live uniform. Sampling settings are not saved in `.ntc` files.

Stochastic sampling needs a temporal reconstruction strategy in your renderer for stable images. The library and demo do not supply a temporal resolve. Its expected reconstructed channel values match trilinear sampling, but shading is nonlinear: averaging shaded frames is not identical to shading averaged material channels.

## UVs and mip selection

The model carries the source texture resolution and its trained mip range. Automatic LOD uses screen-space UV derivatives. Positive `lodBias` selects **finer** mips in this library, so its sign convention may differ from other texture APIs.

An explicit constructor `lodNode` overrides automatic LOD; `setLodBias()` then has no effect on mip selection. This is useful for inspecting one fixed mip:

```ts
import { float } from 'three/tsl';

const fixedMipMaterial = new NTCNodeMaterial(cpuModel, channelClassification, {
  lodNode: float(2),
});
```

The exported affine `uvTransform` maps mesh UVs into the model's local texture space. The runtime uses it automatically. A constructor `uvTransform` can override it. Keep baking and query transforms consistent: changing tiling after training changes where the learned material is sampled, not what was learned.

## Debugging and quality

Use `material.setDebugView('albedo')` to inspect an individual channel, `material.setDebugView('textureUv')` to inspect local UVs, and `material.setDebugView('shaded')` to restore lighting. Debug channel names come from the built-in channel vocabulary and depend on the asset's material properties.

When a model looks wrong, compare the same UV region, mip, lighting, exposure, and sampling mode against its source. Check that the model has trained long enough, that source textures have the expected color space, and that normal maps follow the source material's conventions. Trilinear filtering can hide nearest-sampling aliasing; it cannot recover details that the learned model failed to represent.

## Storage is different from GPU memory

The `.ntc` manifest stores quantized, base64-packed latent arrays and float16-packed decoder parameters. The loader expands these into CPU arrays; the renderer builds feature textures and decoder uniforms. JSON size, decoded payload size, CPU memory, and resident GPU memory are different measurements. A small download is not a promise of equivalent GPU memory usage or faster shading.

For live training previews, `updateFromModel()` uploads compatible model updates in place. Model shape must remain compatible; create a new material when changing architecture. Dispose replaced materials after detaching them from preview meshes.
