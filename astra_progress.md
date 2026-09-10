# NTC quality and correctness progress

Baseline source revision: `d8fc5cb`. Each improvement is committed separately.

## Measurement protocol

`node scripts/ntc-quality.mjs LABEL` runs six fixed WebGPU compression cases:
smooth, checkerboard, and wave material channels, each with learned interpolation
enabled and disabled. Each packs RGB albedo and scalar roughness in [0,1].
Settings: 64x64 source, 16x16 finest grid, 3 levels, 4 features, two 16-wide
hardGELU layers, seed 7, 400 iterations, 2048 samples/batch, uint4 quantization,
and 5% frozen-grid adaptation. Architectural fixes can change parameter count;
those comparisons are not claims of equal bitrate or equal decode cost.

We evaluate every texel of every source mip (64x64 through 1x1) using GPU
inference, both before and after export/reload. Per-mip MSE is the mean squared
error of the four channels. Aggregate MSE weights mips by their texel counts;
PSNR is -10 log10(MSE). The summary averages the six exported-model MSEs before
converting to PSNR. Lower MSE and higher PSNR are better. Raw measurements,
including per-mip errors, are saved under `docs/metrics/`.

These small synthetic fixtures diagnose regressions; they do not establish
quality on the paper's real-material dataset. The source mip target remains
fixed throughout comparisons. We report regressions as well as improvements.
Correctness assertions are separate from these measurements, which have no
arbitrary PSNR pass threshold. GPU/driver differences may affect results.

## Progress

1. Native feature sampling: regression verifies alternating features survive
   LOD 1 within a two-mip band. Decoder GPU suite: 26 passed; TypeScript passed.
2. Quantize taps before interpolation: pending.
3. Independent and complete UV/LOD sampling: pending.
4. Frozen-gradient handling: pending.
5. G0/G1 pairs and feature capacity: pending.
6. Resolution-aware mip bands: pending.
7. Discrete targets and decoded-value filtering: pending.
8. Preset consistency and paper baseline: pending.
9. Bounded noise quantization: pending.
10. Positional encoding, source mip filtering, convergence coverage: pending.

## Measurements

| Revision / change | Exported MSE | PSNR (dB) | MSE change vs previous |
|---|---:|---:|---:|
| baseline | 0.01294998 | 18.8773 | — |
| step1: native features | 0.01176782 | 19.2930 | -9.13% |
