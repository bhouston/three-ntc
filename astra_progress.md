# NTC quality and correctness progress

Baseline source revision: `d8fc5cb`. Each improvement is committed separately.

## Measurement protocol

`node scripts/ntc-quality.mjs LABEL` runs six fixed WebGPU compression cases:
smooth, checkerboard, and wave material channels, each with learned interpolation
enabled and disabled. Each packs RGB albedo and scalar roughness in [0,1].
Settings: 64x64 source, 16x16 finest grid, 3 levels, 4 features, two 16-wide
hardGELU layers, seed 7, 420 total optimizer iterations, 2048 samples/batch, uint4 quantization,
and 5% frozen-grid adaptation. Through step8 the configuration and raw JSON
recorded 400 main iterations, with 20 extra adaptation steps. From step9 the
configured 420 iterations include adaptation (399 main + 21 frozen), and the
cosine rate spans the whole budget. Thus the actual update count stays fixed. Architectural fixes can change parameter count;
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
6. Resolution-aware mip bands: complete. Unit tests cover the paper's
   1024/256 mapping, boundaries, and other resolution ratios. All 25 unit tests
   and TypeScript checks pass. The waves/learned case regresses at this budget;
   the aggregate improves because coarse checker mips now use the right grids.
7. Discrete targets and decoded-value filtering: complete. GPU regressions
   verify exact texel centers, spatial/mip filtering after sigmoid, and nearest
   sampling. Existing 30 decoder and 8 training tests pass. Exact trilinear
   filtering uses eight MLP evaluations with shared weights. The metric samples
   texel centers, so its improvement measures training, not off-center filtering.
8. Preset consistency and paper baseline: complete. Defaults derive from the
   selected profile; hidden width and layer count both survive preset changes.
   A tested opt-in 8/12-feature, 64x64 hardGELU paper profile includes separate
   MLP/latent rates and the full paper budget. Nine training GPU tests and
   27 unit tests pass. Fixed benchmark settings remain unchanged.
9. Bounded noise quantization: complete. GPU tests verify the exact noise
   stream, its mean/variance, disabled noise during adaptation, post-Adam bounds,
   and finite constant-range quantization. All nine training tests pass.
   STE with explicit/auto ranges remains available for comparison.
10. Positional encoding: complete. GPU tests verify an eight-texel period at
   multiple target mips, distinguish the previous four-texel repetition, and
   preserve legacy assets. All 40 decoder/training/PE GPU tests pass.
   The eight-texel option regressed aggregate MSE by 7.16% in step10a, so step10b
   restores grid-cell PE as the default. `positionalEncodingPeriod: 8` remains
   explicit and is selected by the paper profile. This is a measured reason
   to retain the default deviation, rather than assuming paper fidelity wins.
11. Lanczos source mip option and quantitative filter tests: complete. Signed
   DC, odd/rectangular CPU mips, and GPU upload/training/ownership tests pass.
   `mipFilter: "lanczos"` rebuilds square GPU sources; the default preserves
   caller-supplied mips. The paper profile opts in. Fixed compression fixtures
   retain their box source targets, so their compression metric stays unchanged.
12. Longer convergence measurements and final verification: pending.

## Measurements

| Revision / change | Exported MSE | PSNR (dB) | MSE change vs previous |
|---|---:|---:|---:|
| baseline | 0.01294998 | 18.8773 | — |
| step1: native features | 0.01176782 | 19.2930 | -9.13% |
| step2: quantize taps before interpolation | 0.01153805 | 19.3787 | -1.95% |
| step3: independent full-domain sampling | 0.01007776 | 19.9664 | -12.66% |
| step4: exclude frozen latent gradients | 0.01007776 | 19.9664 | 0.00% |
| step5: independent G0/G1 pairs | 0.01003036 | 19.9868 | -0.47% |
| step6: resolution-aware feature mip bands | 0.00852505 | 20.6930 | -15.01% |
| step7: discrete texel training and decoded-value filtering | 0.00621377 | 22.0665 | -27.11% |
| step8: consistent presets and opt-in paper profile | 0.00621377 | 22.0665 | 0.00% |
| step9: bounded noise QAT and in-budget frozen adaptation | 0.00613309 | 22.1232 | -1.30% |
| step10a: eight-texel positional encoding | 0.00657211 | 21.8230 | 7.16% |
| step10b: retain measured default; expose paper PE option | 0.00613309 | 22.1232 | -6.68% |
| step11: optional Lanczos source mipmaps | 0.00613309 | 22.1232 | 0.00% |

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

### step6: resolution-aware feature mip bands

Compared with step5. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00257973 | 25.884 | -30.20% |
| smooth | true | 0.00230994 | 26.364 | 3.56% |
| checker | false | 0.00915303 | 20.384 | -46.59% |
| checker | true | 0.00909203 | 20.413 | -7.91% |
| waves | false | 0.02384627 | 16.226 | -0.94% |
| waves | true | 0.00416928 | 23.799 | 31.38% |

### step7: discrete texel training and decoded-value filtering

Compared with step6. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00242112 | 26.160 | -6.15% |
| smooth | true | 0.00185991 | 27.305 | -19.48% |
| checker | false | 0.00514766 | 22.884 | -43.76% |
| checker | true | 0.00281665 | 25.503 | -69.02% |
| waves | false | 0.02368443 | 16.255 | -0.68% |
| waves | true | 0.00135283 | 28.688 | -67.55% |

### step8: consistent presets and opt-in paper profile

Compared with step7. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00242112 | 26.160 | 0.00% |
| smooth | true | 0.00185991 | 27.305 | 0.00% |
| checker | false | 0.00514766 | 22.884 | 0.00% |
| checker | true | 0.00281665 | 25.503 | 0.00% |
| waves | false | 0.02368443 | 16.255 | 0.00% |
| waves | true | 0.00135283 | 28.688 | 0.00% |

### step9: bounded noise QAT and in-budget frozen adaptation

Compared with step8. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00237807 | 26.238 | -1.78% |
| smooth | true | 0.00154572 | 28.109 | -16.89% |
| checker | false | 0.00553242 | 22.571 | 7.47% |
| checker | true | 0.00219570 | 26.584 | -22.05% |
| waves | false | 0.02387866 | 16.220 | 0.82% |
| waves | true | 0.00126799 | 28.969 | -6.27% |

### step10a: eight-texel positional encoding

Compared with step9. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00237807 | 26.238 | 0.00% |
| smooth | true | 0.00158167 | 28.009 | 2.33% |
| checker | false | 0.00553242 | 22.571 | 0.00% |
| checker | true | 0.00236230 | 26.267 | 7.59% |
| waves | false | 0.02387866 | 16.220 | 0.00% |
| waves | true | 0.00369953 | 24.319 | 191.76% |

### step10b: retain measured default; expose paper PE option

Compared with step10a. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00237807 | 26.238 | 0.00% |
| smooth | true | 0.00154572 | 28.109 | -2.27% |
| checker | false | 0.00553242 | 22.571 | 0.00% |
| checker | true | 0.00219570 | 26.584 | -7.05% |
| waves | false | 0.02387866 | 16.220 | 0.00% |
| waves | true | 0.00126799 | 28.969 | -65.73% |

### step11: optional Lanczos source mipmaps

Compared with step10b. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00237807 | 26.238 | 0.00% |
| smooth | true | 0.00154572 | 28.109 | 0.00% |
| checker | false | 0.00553242 | 22.571 | 0.00% |
| checker | true | 0.00219570 | 26.584 | 0.00% |
| waves | false | 0.02387866 | 16.220 | 0.00% |
| waves | true | 0.00126799 | 28.969 | 0.00% |

### Lanczos filter measurements

`node scripts/ntc-mip-quality.mjs` compares production Lanczos-3 and box
downsampling against analytically ideal low-pass sinusoidal targets. This
is a filter-quality measurement, separate from compression MSE.

| Signal (cycles/source texel) | Box MSE | Lanczos MSE |
|---|---:|---:|
| 0.125 (passband) | 0.00289716 | 0.00006849 |
| 0.375 (above new Nyquist) | 0.07322330 | 0.00005563 |

Lanczos preserves negative lobes and signed physical channels. It can ring;
these two frequency tests do not establish superiority for every material.

### Physical mip bounds

A final edge-case check found `ceil(log2(size))` requested a nonexistent mip
for non-power-of-two sources. Both model layouts now use floor, and the loader
accepts a valid maxLod of zero. CPU/roundtrip checks cover sizes 1, 3, 6, 1000;
GPU training on 1x1 and 6x6 sources produces finite losses. Power-of-two
benchmark geometry is unchanged.

### Final loss and frozen-grid regression

Training now returns a finite final batch loss even when no progress callback
is installed. A GPU regression also observes both G0 and G1 remaining exactly
unchanged over multiple adaptation callbacks while MLP parameters change.
All 10 training GPU tests pass. This adds readback/reporting only and does not
change model updates or the reconstruction metric.
