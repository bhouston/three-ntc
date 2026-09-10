# NTC quality and correctness progress

Baseline source revision: `d8fc5cb`. Each improvement is committed separately.

The equal-update material-channel benchmark improved exported MSE from
**0.01085313 to 0.00530139 (-51.15%)**. Every one of its six cases improved.
The step-by-step linear diagnostic improved from **0.01294998 to 0.00613309
(-52.64%)**. An additional 3x-budget run reaches **0.00429097**; that is a
convergence result, not another equal-budget implementation improvement.

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

The step-by-step suite uses linear outputs to isolate the grid, training, and
codec behavior. The separate `physical-baseline` / `physical-final` comparison
trains and evaluates the channels with their actual sigmoid activations, as
used by material reconstruction at texel centers. Both use 420 total updates.
The original revision was rerun with that harness; these are measured baseline
values, not estimates from the linear-output results.

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
12. Longer convergence measurements and final verification: complete. Increasing
   the same diagnostic budget from 420 to 1260 updates lowers aggregate MSE
   another 30.04%; all six cases improve. This is still one seed and synthetic
   data, not a substitute for the paper's full-budget real-material evaluation.

## Measurements

| Revision / change | Exported MSE | PSNR (dB) | MSE change vs stated comparison |
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
| physical-baseline (separate sigmoid benchmark) | 0.01085313 | 19.6444 | — |
| physical-final: physical channel activations (separate benchmark) | 0.00530139 | 22.7561 | -51.15% |
| convergence: 1260 updates (3x budget, separate comparison) | 0.00429097 | 23.6744 | -30.04% |

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

### physical-final: physical channel activations (separate benchmark)

Compared with physical-baseline. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00069507 | 31.580 | -82.64% |
| smooth | true | 0.00055200 | 32.581 | -79.67% |
| checker | false | 0.00348914 | 24.573 | -84.77% |
| checker | true | 0.00329273 | 24.824 | -61.55% |
| waves | false | 0.02321669 | 16.342 | -4.47% |
| waves | true | 0.00056270 | 32.497 | -78.58% |

### convergence: 1260 updates (3x budget, separate comparison)

Compared with step11. Six quality cases passed (finite error only; no PSNR threshold).

| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |
|---|---|---:|---:|---:|
| smooth | false | 0.00063067 | 32.002 | -73.48% |
| smooth | true | 0.00034864 | 34.576 | -77.44% |
| checker | false | 0.00087724 | 30.569 | -84.14% |
| checker | true | 0.00049994 | 33.011 | -77.23% |
| waves | false | 0.02306910 | 16.370 | -3.39% |
| waves | true | 0.00032023 | 34.945 | -74.75% |

## Reproduction and validation

```sh
node scripts/ntc-quality.mjs current
NTC_BENCH_PHYSICAL=1 node scripts/ntc-quality.mjs physical-current
NTC_BENCH_ITERATIONS=1260 node scripts/ntc-quality.mjs convergence-current
NTC_BENCH_PE_PERIOD=8 node scripts/ntc-quality.mjs paper-pe-current
node scripts/ntc-mip-quality.mjs
pnpm test
pnpm test:gpu
pnpm -r build
```

The quality runner snapshots source files before running, so edits during a
measurement cannot change its implementation. An optional third argument points
to a separately prepared checkout; the original revision was measured that way
with the same fixtures/readback harness. All raw per-case/per-mip results are
committed here. The full paper training budget has **not** been run.

Final validation: 32 unit tests and TypeScript pass. The complete GPU run passed
56 cases; subsequent focused runs passed all 13 affected cases, including three
new regressions (59 GPU cases in the final suite). All package, CLI, and website
production builds pass. Lint exits successfully with pre-existing warnings;
the website build retains its bundle-size warning.

API changes: native grid textures are required by the raw decoder; G1 grids are
independent and can have a different channel count; `iterations` includes frozen
adaptation; `setInterpolation` filters decoded values; noise QAT is the default
when a quantized mode is selected without an explicit range. Legacy loaded
assets retain their former PE phase and shared-coarsest G1 fallback. The paper
profile is an explicit, expensive configuration, not a claim of matching a
published bitrate. Exact runtime trilinear filtering costs eight MLP evaluations.

## Runtime shader expansion (2026-09-10)

The eight reconstructed taps now share a shader loop, and dense layers use named
shader loops over mat4/vec4 blocks with uniform bounds. Static expansion of the
matrix sums contributed substantial first-use driver latency; `.toVar()` alone
and a real function around the MLP did not resolve it in the measured workload.
See [the investigation](tsl_performance_regression_cause_and_fix.md).

On Chromium **SwiftShader**, 32-wide first render/completed readback decreased from
11,113 ms to 663 ms; the 64-wide case completed in 736 ms. CPU-reference MSE stayed
3.4234e-11 and 5.1345e-11 respectively. Conservative private storage is 4,388/4,900
bytes. These are individual timing runs, not a guaranteed speed ratio. All 63 GPU
tests, all 32 unit tests, TypeScript, and production builds passed with the pending
preview update included. Runtime-only GPU coverage comprises 35 passing cases.

This fixes the storage-limit/expansion problem but does **not** establish that the
user's viewer slowdown is resolved: the user still reports about 6 fps even without
training. Investigation continues against the shipped brick and viewer lighting.

## Preserve the training preview material (2026-09-10)

Progress updates upload existing latent textures and weight uniforms in place,
without rebuilding the material. A browser test verifies changed pixel values,
unchanged texture identities/material version, and no new shader modules. A
separate two-device profile trains five steps with updates at 1 and 5: no WebGPU
errors, two shader modules total, loss 0.29954 -> 0.23010. SwiftShader completed in
3.672 s (44.4 ms max heartbeat gap); Apple Metal completed in 1.066 s, with steady
256px frames at 5.5–6.3 ms. The runtime-only viewer issue remains a separate task.

## Small-network specialization and standalone brick profile (2026-09-10)

Retained static matrix operations for layers no larger than four input/output
vec4 blocks, avoiding indexed-array loops for the shipped 5–8–8–10 brick network.
Large networks retain compact loops. Initial Chromium/Metal HDR sphere timings
improved from 5.6–9.1 ms to 3.5–4.2 ms at 1024px; later runs varied to 9.9–11.3 ms.
Raw CPU-reference MSE was 7.6800803e-6 for both the static and loop implementations.
SwiftShader MSE was 1.7720459e-6. No training math or filtering semantics changed.

64 Chromium/Metal GPU cases, 36 SwiftShader runtime cases, 32 unit cases, and
production builds passed. The user confirmed the remaining ~6 fps is in **Safari**;
Chromium results do not resolve that Safari regression. Conditional feature-level
sampling did not consistently help and was discarded.

## Installed Safari test runner (2026-09-10)

Added `pnpm test:gpu:safari` using Vitest's WebdriverIO provider and Apple's native
`safaridriver`, with visible, serial browser execution. It runs the same GPU tests
as Chromium, including the standalone brick and two-update trainer profiles.

The Safari 27 driver was reached, but session creation failed with its explicit
requirement to enable **Safari Settings → Developer → Allow remote automation**.
No Safari shader test has executed yet; this is a blocked launch, not a pass.
The setting is pending user action. After installing the provider, TypeScript,
32 unit tests, and the Chromium/Metal brick profile passed. The latter reported
raw CPU-reference MSE 7.6800803e-6 and no WebGPU errors.

## Disable MSAA across website viewers (2026-09-10)

The actual `NTCViewer` component reproduces a sustained WebKit slowdown with 4×
MSAA that the offscreen tests omitted. At 1024×768 CSS pixels and DPR 2, the final
five measured animation intervals were 50–60 ms and GPU completion delays were
650–657 ms (including accumulated queued work). Chromium/Metal with MSAA stayed
at 16.4–16.9 ms intervals and 10.8–12.3 ms completion delays.

Per user instruction, disabled MSAA without adding FXAA in the standalone viewer,
trainer comparison preview, and gallery; the shared training/baking renderer also
explicitly disables it. The same WebKit Retina
viewer now reports 16–17 ms intervals and 10–11 ms completion delays. Its test
verifies disabled MSAA, 25 completed frames, and no WebGPU errors for standalone,
trainer comparison, and six-sphere gallery cases. The trainer case uses a simple
teacher material and does not run training concurrently. Raw measurements:
[docs/metrics/runtime-msaa.json](docs/metrics/runtime-msaa.json). The model decoder
was not changed by this mitigation. All 32 unit tests and TypeScript passed.

Native Safari automation remains distinct: the user enabled the required setting,
but Safari 27's driver subsequently timed out creating a session, even with a
direct WebDriver request. A browser restart was requested. The successful Retina
measurements are Playwright WebKit 26.6, not a claim of a completed Safari 27 test.


WebKit verification after applying this across the website (medians of 20 frames
after five warmup frames):

| Component case | Frame interval | Submit-to-completion delay |
| --- | --- | --- |
| single | 17 ms | 11 ms |
| trainer | 17 ms | 8 ms |
| grid | 16.5 ms | 5.5 ms |

All three cases passed in both WebKit and Chromium/Metal at DPR 2. TypeScript,
all 32 unit tests, and the website production build passed.

## Expanded browser performance matrix (2026-09-10)

Committed the concise cause summary in
[docs/viewer_performance_summary.md](docs/viewer_performance_summary.md).
Added a serial `pnpm profile:browser chromium|webkit|safari` runner that saves JSON
metrics and complete logs, including failures. See
[the workload and reproduction guide](docs/browser_performance_tests.md).

Actual website components now run for 120 frames each at two CSS sizes and DPR 1/2
in Playwright. Metrics include first-frame time, raw steady-frame intervals, GPU
completion delays, heartbeat gaps, and mean/p50/p95/p99/max summaries. Five-step
training also renders an animated physical preview concurrently on a separate
renderer, retaining loss, finite-pixel, and shader-reuse checks.

Chromium/Metal and WebKit each passed 16 cases (eight per DPR). At 1024×768 CSS,
DPR 2, steady-state frame interval medians/p95 were:

| Viewer | Chromium median / p95 | WebKit median / p95 |
| --- | --- | --- |
| Standalone | 16.6 / 17.3 ms | 17 / 18 ms |
| Trainer comparison | 16.7 / 17.5 ms | 17 / 18 ms |
| Gallery | 16.7 / 17.4 ms | 17 / 18 ms |

The first training profiles exposed startup stalls: maximum heartbeat gaps were
379.4 ms in Chromium and 512 ms in WebKit. Subsequent profiles reported 39.4 ms
and 74 ms respectively. These are observations from separate runs with potentially
warm driver caches, not evidence that DPR improves training (its offscreen preview
has fixed resolution). Both engines retained shipped-brick raw CPU-reference MSE
7.6800803e-6; training loss remained 0.299537 → 0.230104. No WebGPU errors occurred.

Native Safari was retried after the user enabled automation, including through the
new runner. Its session still timed out before tests executed; this is recorded as
a failed launch with no timings. No new Safari settings were changed. Summary
measurements are in [browser-profile-baseline.json](docs/metrics/browser-profile-baseline.json).
TypeScript and all 34 unit tests passed, including percentile/tail-stall validation.

## Correct the trainer's model-size estimate (2026-09-10)

The size card ignored the quantization setting and always estimated uint8 storage.
Positional encoding and dual grid were already included in its dependencies, but
rounded byte totals did not clearly expose small decoder changes. Replaced the
single summary line with encoded-payload and runtime estimates showing exact bytes,
plus decoder input/parameter counts and work per decoder evaluation. The estimate
reflects the current form settings, not the previously trained preview model.

The packed payload now follows 8/4/2-bit export. The existing export behavior for
training quantization `none` is still uint8; the card states this explicitly. Runtime
storage does not shrink with export quantization because grids are expanded to FP16.
Labels distinguish payload estimates from JSON/base64 file size and exclude GPU
padding and training buffers.

Tests build and export real CPU models for all 16 combinations of positional
encoding, dual grid, and quantization mode. Estimated packed bytes match the decoded
base64 payload lengths exactly (zero byte error in every case). For the small
16px/two-level, 16-wide/one-hidden-layer, three-output fixture:

| Setting (PE and dual on unless stated) | Payload bytes |
| --- | --- |
| 8-bit / training quantization off | 2,550 |
| 4-bit | 1,870 |
| 2-bit | 1,530 |
| 8-bit, PE off | 1,782 |
| 8-bit, dual grid off | 2,150 |

Previously the 4-bit and 2-bit estimates overstated this fixture by 680 and 1,020
bytes. Browser tests drive the same TanStack form subscription as the trainer and
verify the visible card updates and restores its values. WebKit and Chromium tests,
all 51 unit tests, TypeScript, and the website production build passed. This change
only affects estimates and display; training and material reconstruction are unchanged.
