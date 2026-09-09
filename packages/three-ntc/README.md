# three-ntc

[![npm version](https://img.shields.io/npm/v/three-ntc.svg)](https://www.npmjs.com/package/three-ntc)
[![npm downloads](https://img.shields.io/npm/dm/three-ntc.svg)](https://www.npmjs.com/package/three-ntc)
[![ci](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-three--ntc.ben3d.ca-blue)](https://three-ntc.ben3d.ca)

Neural Texture Compression runtime for three.js — load `.ntc` files and get back a standard
`MeshPhysicalNodeMaterial` (via TSL) that decodes a compact grid + MLP model on the GPU.

![Eight NTC materials rendered live in the three-ntc grid viewer](../../docs/images/ntc-grid-viewer.png)

An implementation of NVIDIA's 2023 [Neural Texture Compression](https://research.nvidia.com/labs/rtr/neural_texture_compression/)
paper: NTC jointly fits a material's whole texture stack — albedo, normal, roughness, metalness,
and more — into a single grid + MLP model instead of one texture per channel. Every material
pictured above is about 93KB total, often smaller than a single texture from the original
material, for a fraction of the memory and download cost of standard PBR textures.

Try it live at **[three-ntc.ben3d.ca](https://three-ntc.ben3d.ca)**.

`.ntc` models are trained with the companion [`three-ntc-trainer`](https://www.npmjs.com/package/three-ntc-trainer) package.

## Install

```bash
npm install three-ntc
```

## Usage

```js
import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

const loader = new NTCLoader();
const { cpuModel, channelClassification } = await loader.loadAsync('gold.ntc');

const material = new NTCNodeMaterial(cpuModel, channelClassification);
const mesh = new THREE.Mesh(geometry, material);
```

`NTCNodeMaterial` extends three.js's `MeshPhysicalNodeMaterial`, so it drops straight into an
existing WebGPU renderer scene alongside ordinary materials.
