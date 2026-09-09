# three-ntc

Neural Texture Compression for three.js — encode PBR material texture sets into a compact
grid + MLP model that decodes on the GPU via TSL, and load the result as a standard three.js
node material.

## Packages

- [`packages/three-ntc`](packages/three-ntc) — runtime: `NTCLoader` + `NTCNodeMaterial`. This is the
  package to depend on if you just want to load `.ntc` files in your three.js app.
- [`packages/three-ntc-trainer`](packages/three-ntc-trainer) — trains a `.ntc` model from a MaterialX
  material (or baked textures) on the GPU, and exports it via `NTCExporter`.
- [`packages/website`](packages/website) — demo site (viewer + trainer UI), TanStack Start + Router +
  Tailwind + shadcn/ui.
- [`packages/cli`](packages/cli) — command-line trainer (stub).

## Development

```bash
corepack enable
pnpm install
pnpm dev
```
