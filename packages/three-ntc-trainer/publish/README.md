# three-ntc-trainer

[![npm version](https://img.shields.io/npm/v/three-ntc-trainer.svg)](https://www.npmjs.com/package/three-ntc-trainer)
[![npm downloads](https://img.shields.io/npm/dm/three-ntc-trainer.svg)](https://www.npmjs.com/package/three-ntc-trainer)
[![ci](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-three--ntc.ben3d.ca-blue)](https://three-ntc.ben3d.ca)

GPU trainer that bakes a three.js material (MaterialX or plain baked textures) into a compact
`.ntc` Neural Texture Compression model — a shared multiresolution latent grid plus MLP decoder,
jointly fit against the material's PBR channels — and exports it for the
[`three-ntc`](https://www.npmjs.com/package/three-ntc) runtime to load.

Try the trainer live at **[three-ntc.ben3d.ca](https://three-ntc.ben3d.ca)**.

## Install

```bash
npm install three-ntc-trainer
```

## Usage

```js
import { fitNTCMaterial, NTCExporter } from 'three-ntc-trainer';

// `material` is any three.js node material (e.g. loaded via MaterialXLoader).
const trainedMaterial = await fitNTCMaterial(renderer, material, {
  onProgress: ({ step, loss }) => console.log(step, loss),
});

const exporter = new NTCExporter();
const manifest = exporter.parse(trainedMaterial, { name: 'Gold' });
const blob = new Blob([JSON.stringify(manifest)], { type: 'application/json' });
```

The resulting manifest is the `.ntc` file `NTCLoader` (from `three-ntc`) reads back.

For finer control over training (grid resolution, MLP shape, step count, ...), construct
`NTCTrainer` directly instead of using the `fitNTCMaterial` convenience wrapper.
