<p align="center">
  <a href="https://three-ntc.ben3d.ca" target="_blank" rel="noopener noreferrer">
    <img alt="three-ntc logo" src="https://raw.githubusercontent.com/bhouston/three-ntc/main/packages/docs/static/img/icon-512.png" height="96">
  </a>
</p>
<br/>
<p align="center">
  <a href="https://npmjs.com/package/three-ntc-trainer"><img src="https://img.shields.io/npm/v/three-ntc-trainer.svg" alt="npm package"></a>
  <a href="https://www.npmtrends.com/three-ntc-trainer"><img src="https://img.shields.io/npm/dw/three-ntc-trainer.svg" alt="npm downloads"></a>
  <a href="https://codecov.io/gh/bhouston/three-ntc"><img src="https://codecov.io/gh/bhouston/three-ntc/branch/main/graph/badge.svg" alt="coverage"></a>
  <a href="https://discord.gg/fwupDN493R"><img src="https://img.shields.io/badge/chat-discord-blue?style=flat&logo=discord" alt="discord chat"></a>
  <a href="https://three-ntc.ben3d.ca/docs/getting-started/"><img src="https://img.shields.io/badge/docs-read-blue?style=flat" alt="documentation"></a>
</p>
<br/>

# three-ntc-trainer

> Train Neural Texture Compression models for three.js on the GPU.

- 🏋️ GPU Training in the Browser
- 🧪 MaterialX or Baked Texture Input
- 🗜️ Compact `.ntc` Export
- 🔧 Configurable Grid and MLP

Bakes a three.js material into a compact `.ntc` model, a shared multiresolution latent grid plus MLP decoder jointly fit against the material PBR channels, and exports it for the [`three-ntc`](https://www.npmjs.com/package/three-ntc) runtime to load.

[Read the Docs to Learn More](https://three-ntc.ben3d.ca/docs/getting-started/).

![Eight NTC materials rendered live in the three-ntc grid viewer](https://raw.githubusercontent.com/bhouston/three-ntc/main/docs/images/ntc-grid-viewer.png)

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

## Contribution

See [Contributing Guide](https://github.com/bhouston/three-ntc/blob/main/CONTRIBUTING.md).

## License

MIT

## Author

[Ben Houston](https://ben3d.ca), Sponsored by [Land of Assets](https://landofassets.com).
