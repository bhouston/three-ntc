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
2. Quantize taps before interpolation: complete. Training GPU suite: 8 passed.
   Fixed-coordinate uint2 test verifies both bilinear inputs remain 0.4.
3. Independent and complete UV/LOD sampling: complete. Distribution test passed;
   CPU/GPU stream parity and training suite: 9 GPU tests passed.
4. Frozen-gradient handling: complete. Frozen-gradient regression and 8 training
   GPU tests pass. The benchmark is unchanged: clipping did not engage on these
   short adaptation runs, so this fix prevents a failure rather than lowering
   their measured error.
5. G0/G1 pairs and feature capacity: complete. Independent half-resolution G1
   grids and 8/12-channel support pass CPU layout/roundtrip and GPU decoder tests.
   This adds latent capacity; the comparison is not at equal bitrate.
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
| step2: quantize taps before interpolation | 0.01153805 | 19.3787 | -1.95% |

| step3: independent full-domain sampling | 0.01007776 | 19.9664 | -12.66% |

| step4: exclude frozen latent gradients | 0.01007776 | 19.9664 | 0.00% |

| step5: independent G0/G1 pairs | 0.01003036 | 19.9868 | -0.47% |

## Per-change details

### step2: quantize taps before interpolation

Compared with step1. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00416424 | 23.805 | 4.23% |
| smooth | true | 0.00620372 | 22.073 | 0.40% |
| checker | false | 0.01665406 | 17.785 | -6.51% |
| checker | true | 0.01119423 | 19.510 | -2.66% |
| waves | false | 0.02449993 | 16.108 | -0.34% |
| waves | true | 0.00651213 | 21.863 | -0.34% |

### step3: independent full-domain sampling

Compared with step2. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00360238 | 24.434 | -13.49% |
| smooth | true | 0.00351545 | 24.540 | -43.33% |
| checker | false | 0.01639537 | 17.853 | -1.55% |
| checker | true | 0.00949695 | 20.224 | -15.16% |
| waves | false | 0.02396619 | 16.204 | -2.18% |
| waves | true | 0.00349023 | 24.571 | -46.40% |

### step4: exclude frozen latent gradients

Compared with step3. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00360238 | 24.434 | 0.00% |
| smooth | true | 0.00351545 | 24.540 | 0.00% |
| checker | false | 0.01639537 | 17.853 | 0.00% |
| checker | true | 0.00949695 | 20.224 | 0.00% |
| waves | false | 0.02396619 | 16.204 | 0.00% |
| waves | true | 0.00349023 | 24.571 | 0.00% |

### step5: independent G0/G1 pairs

Compared with step4. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00369585 | 24.323 | 2.59% |
| smooth | true | 0.00223058 | 26.516 | -36.55% |
| checker | false | 0.01713714 | 17.661 | 4.52% |
| checker | true | 0.00987279 | 20.056 | 3.96% |
| waves | false | 0.02407228 | 16.185 | 0.44% |
| waves | true | 0.00317351 | 24.985 | -9.07% |
