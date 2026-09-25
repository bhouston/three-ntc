# three-ntc

[![npm version](https://img.shields.io/npm/v/three-ntc.svg)](https://www.npmjs.com/package/three-ntc)
[![ci](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml)
[![Unit coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbhouston%2Fthree-ntc%2Fcoverage-badge%2Fcoverage.json)](https://github.com/bhouston/three-ntc/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/demo-three--ntc.ben3d.ca-blue)](https://three-ntc.ben3d.ca)

Neural Texture Compression for three.js — encode PBR material texture sets into a compact
grid + MLP model that decodes on the GPU via TSL, and load the result as a standard three.js
node material.

![Eight NTC materials rendered live in the three-ntc grid viewer](docs/images/ntc-grid-viewer.png)

An implementation of NVIDIA's 2023 [Neural Texture Compression](https://research.nvidia.com/labs/rtr/neural_texture_compression/)
paper: instead of shipping a full albedo/normal/roughness/metalness texture stack per material,
NTC jointly fits all of those channels into one small grid + MLP model. Every material pictured
above — all its textures combined — is about 93KB, often smaller than a single texture from the
original material, so scenes can carry far richer materials without blowing up download size.

Try it live at **[three-ntc.ben3d.ca](https://three-ntc.ben3d.ca)**.

## Usage

```js
import { NTCLoader, NTCNodeMaterial } from 'three-ntc';

const loader = new NTCLoader();
const { cpuModel, channelClassification } = await loader.loadAsync('gold.ntc');

const material = new NTCNodeMaterial(cpuModel, channelClassification);
const mesh = new THREE.Mesh(geometry, material);
```

## Packages

- [`packages/three-ntc`](packages/three-ntc) — runtime: `NTCLoader` + `NTCNodeMaterial`. This is the
  package to depend on if you just want to load `.ntc` files in your three.js app. See its
  [README](packages/three-ntc/README.md) for install/usage.
- [`packages/three-ntc-trainer`](packages/three-ntc-trainer) — trains a `.ntc` model from a MaterialX
  material (or baked textures) on the GPU, and exports it via `NTCExporter`. See its
  [README](packages/three-ntc-trainer/README.md) for install/usage.
- [`packages/website`](packages/website) — demo site (viewer + trainer UI), TanStack Start + Router +
  Tailwind + shadcn/ui.
- [`packages/cli`](packages/cli) — command-line trainer (stub).

## Development

```bash
corepack enable
pnpm install
pnpm dev
```

## Testing and quality gates

- `pnpm test` runs `tsc` then `vitest run --project unit --coverage`. Unit coverage
  includes all runtime and trainer TypeScript source, including GPU paths that unit
  tests don't execute. Gates are statements 19%, branches 13%, functions 18%, lines 19%,
  based on measured coverage rather than an artificial promise — raise them as CPU/GPU
  coverage improves, and don't lower them or exclude files just to pass.
- `pnpm test:gpu` runs the WebGPU suite in headless Chromium (`pnpm test:gpu:safari` for
  Safari). Hosted runners have no GPU, so CI falls back to SwiftShader software Vulkan;
  that job is advisory (`continue-on-error`) until it proves stable.
- `pnpm size` gates minified gzip ESM bundle size with `three` external: 11,000 bytes for
  `three-ntc`, 52,000 bytes for `three-ntc-trainer`. Explain intentional limit changes in
  the PR.
- `pnpm audit --prod --audit-level high` blocks CI on high/critical production
  dependency advisories. The full development audit currently reports two high-severity
  `extract-zip@2.0.1` advisories via WebdriverIO/Puppeteer with no patched registry
  release yet, so it stays advisory (`continue-on-error`) until one ships.
- `pnpm release:check` tests release-version analysis and packaging without publishing.

## Contributing and releases

See [CONTRIBUTING.md](CONTRIBUTING.md) for the shared issue, branch, PR and commit
workflow, [release setup](docs/releasing.md) for npm trusted publishing and the
two-package (runtime + trainer) release config, and [SECURITY.md](SECURITY.md) for
private vulnerability reports.
