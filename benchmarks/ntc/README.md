# MaterialX feature-contribution benchmark

This experiment measures the seven requested materials with **4,000 complete optimizer iterations per result**, export/reload validation, full mip-chain evaluation, and three paired initialization seeds. Smoke runs cannot be included in production analysis.

```sh
# Full experiment: controlled comparisons plus native-default profiles.
node benchmarks/ntc/run.mjs --label materialx-4000

# Resume the same immutable source snapshot, skipping complete and failed cases.
node benchmarks/ntc/run.mjs --label materialx-4000

# Retry failed cases explicitly, or regenerate the dashboard.
node benchmarks/ntc/run.mjs --label materialx-4000 --retry-failed
node benchmarks/ntc/run.mjs --label materialx-4000 --report-only

# Review the full plan without starting training.
node benchmarks/ntc/run.mjs --label planned-study --plan-only

# A small integration check, deliberately marked smoke.
node benchmarks/ntc/run.mjs --label smoke --materials uv_grid --variants reference --iterations 8 --resolution 64 --seeds 1

# Limit a session to a number of completed/attempted cases; resume later.
node benchmarks/ntc/run.mjs --label materialx-4000 --limit 5
```

Results are under `output/benchmarks/<label>/`: `report.html`, `analysis.md`, `summary.json`, `runs.csv`, `contributions.csv`, `per-property-mip.csv`, individual raw JSON results, representative preview PNGs, and per-case logs. Open `report.html` locally; it requires no server or CDN. Refresh it while runs complete. Output is ignored by git because source snapshots and raw measurements are large.

## Experimental design

There are 756 cases: 36 variants × 7 materials × 3 seeds. The deterministic randomized order reduces thermal/order bias. Native runs are scheduled last because their sample budgets are much larger. GPU jobs run sequentially.

- **18 crossed variants:** encoding off/grid-cell/eight-texel tile × G0 bilinear/four-tap concatenation × uint2/uint4/uint8.
- **12 additional variants:** ReLU, hidden widths 16 and 64, ReLU with width 16 (activation/width interaction), G1 off, base resolutions 128 and 512, one and two feature levels, mip spacing 1 and 3, and G0/G1 channel counts 8/12.
- **3 equal-budget profiles:** actual library, website, and paper profile settings with batch size 4096 and 4000 iterations. These compare configuration bundles, not individual feature effects.
- **3 native-budget profiles:** retain native batch sizes (4096 / 8192 / 524288) and all profile settings except the user-requested 4000-iteration budget.

The controlled reference has four-channel G0/G1, G0 resolutions 256/64/16/4, two 32-neuron hardGELU layers, G1 enabled, four-tap G0 sampling, eight-texel PE, and uint4 quantization. The source is baked at 1024². “Hidden 16” means **two hidden layers of width 16**, not sixteen hidden layers.

The quantizers are the repository's **unsigned packed uint2/uint4/uint8 codecs** (the requested 2/4/8-bit comparison). Quantization bin centers can represent signed latent values; these are not signed integer storage formats.

Encoding and sampling are now independently configurable with `g0Sampling`. Omitting it preserves the previous coupling to `positionalEncoding`, so existing callers and assets retain behavior. Export/load and CPU/GPU paths carry the independent choice.

The four-tap G0 noise-QAT discrepancy was corrected after the main-profile study, with a focused GPU regression test. The controlled feature matrix still explicitly uses STE on both grids, preventing a sampling change from also changing the quantization method. Historical snapshots preserve their original behavior; never mix results from different snapshots. New runs save exported models and gzip (level 9) / Brotli (quality 5) variants in `models/`, recording actual transport byte counts. Source auditing now rejects an entirely constant bake before training; numerically constant properties within a varying material remain included in the legacy aggregate metric and are labeled in `sourceAudit`.

## Training and memory

Every quantized run includes the trainer's final 5% frozen-grid adaptation within the 4000 iterations: 3800 joint updates and 200 decoder-only updates. This follows the current library's iteration accounting. Unquantized runs use all 4000 joint updates, then the existing uint8 exporter.

The native paper profile entails 2,097,152,000 samples per material and seed. Across its 21 runs that is about 44 billion samples, so completion can take substantially longer than the controlled sweep. The implementation uses `microBatchSize:4096`, distinct global sample indices, accumulated gradients, and **one** Adam update per logical batch. This avoids a several-hundred-MiB activation buffer without reducing the logical batch. A GPU test compares full and chunked batches numerically. Chunking requires floating accumulation; unsupported devices fail explicitly instead of silently accumulating an overflowing fixed-point sum.

Progress readback runs every 128 iterations for controlled runs and every 4 for native runs; this affects measurement overhead, not optimizer updates. Training time includes compilation, readback, and paper-profile source mip rebuilding. `evaluationMs` includes compilation and complete reference/model readback and is **not** a steady-state inference-latency benchmark. Do not compare it to the renderer performance suite's frame times.

## Quality metrics and interpretation

The primary measure is **exported** reconstruction PSNR from physical-range-normalized squared errors over every active scalar and every texel of the complete mip chain. Signed normal/anisotropy components have range 2; other components use range 1. HDR values are not clipped for metrics: their PSNR uses a unit reference. Padding components and properties classified as constant by the material classifier are excluded. Numerically constant outputs that the classifier retained are still included in this legacy aggregate metric; `sourceAudit` labels them explicitly.

Normal properties additionally record reconstructed angular mean/RMS/max error and the fraction of predicted offsets outside the unit disk in `normal-errors.csv`.

Also recorded: mip-0 PSNR, equal-mip PSNR, per-property/per-component/per-mip MSE, normalized MSE, MAE, maximum absolute error, and explicitly named **block SSIM** (disjoint 8×8 windows, population statistics). It is not Gaussian-window SSIM or LPIPS. Exact zero error has a null PSNR in JSON (mathematically +infinity), and is excluded from finite-dB contrasts.

Every model is evaluated against the same box-filtered source mip chain. The native/equal-budget paper profiles train with Lanczos, so those bundle comparisons include the different source filter. Controlled feature comparisons all use the same source filter. UV transforms are inferred and baked in local space exactly as in the website trainer. The PNG panels show one representative channel, not a lighting-dependent rendered material; all active properties are measured numerically. This study measures material-map fidelity, not perceptual similarity under every lighting/view configuration.

The dashboard reports:

- Conditional changes relative to the controlled reference.
- Factorial marginal effects averaged across all matching contexts within each seed.
- Paired-seed mean ΔPSNR and 95% Student-t intervals. With three seeds, intervals will often be broad. Texels and repeated contexts are not counted as independent experiments.

Positive ΔdB means improved reconstruction. Effects are **not additive percentages of total quality**: parameters interact and changing channels/grid dimensions changes capacity and storage. Compare quality with latent payload, exported JSON bytes, runtime padded-grid bytes, and training time. Native profiles have unequal sample budgets and must not be presented as isolated feature effects. Exploratory confidence intervals are not adjusted for multiple comparisons. Findings are restricted to these seven materials and the 4000-step budget; they do not establish fully converged quality.

## Reproducibility and failure handling

The runner snapshots source, tests, fixtures, resource images, package manifests and lockfile before training, records SHA-256 hashes and the git revision, and resumes against that snapshot. Dependencies are symlinked to the current installation; keep the installation fixed for the run. Adapter identity, actual gradient precision, and hashes of every baked reference mip are recorded per result. Reference hashes must agree for a paired comparison. Comparisons with different adapter identities/gradient precision are excluded. Each GPU run has its own browser process, and no GPU runs are parallelized.

Completed runs are atomically recorded; failed cases are recorded with a log path and never included as quality results. A run lock prevents concurrent drivers. SIGINT/SIGTERM requests stop after the current case. If a process is killed, check the PID in `runner.lock/owner.json` before removing that stale directory. Resume with the label only; settings cannot silently change within a run. Use a new label for changed source, defaults, or experimental parameters.

Validation:

```sh
node --test benchmarks/ntc/design.test.mjs
pnpm tsc
pnpm exec vitest run --project unit
pnpm exec vitest run --project gpu packages/three-ntc-trainer/src/NTCAblation.gpu.test.ts
```

## Windows / NVIDIA

Use native Windows with a recent NVIDIA driver and Node.js 24 or newer. Extract the portable source ZIP into a short path such as `C:\ntc-benchmark` (outside OneDrive). The archive includes `benchmark-source.json`, so a Git checkout is not required. It includes the current benchmark changes and source assets; it does not include Mac dependencies or measurements.

From PowerShell in the extracted folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\benchmarks\ntc\run-windows.ps1
```

The launcher installs the pinned pnpm dependencies and Playwright Chromium, checks that a small run uses an NVIDIA adapter with floating accumulation, and starts the complete 756-case experiment. Directory junctions avoid requiring Windows Developer Mode/admin rights for snapshot links. Chromium uses its native Windows backend rather than forcing Metal or Vulkan. No CUDA toolkit is required: the trainer uses browser WebGPU.

Use `-SmokeOnly` for installation and GPU validation only, or `-Label my-other-run` for a new independent experiment. Run the same launcher again to resume the existing label. Keep the terminal open and disable automatic sleep for the long run. To inspect progress, open `output\benchmarks\rtx3060ti-4000\report.html` and refresh after runs finish.

Start a fresh Windows result set. Do not copy the Mac plan/results into it: platform/backend differences require separate runs, and native timing comparisons cannot mix GPUs. To return results, copy `plan.json`, `summary.json`, `analysis.md`, `report.html`, CSVs, `results/`, `previews/`, and `logs/` from the Windows run directory. Exclude `snapshot/`, which contains machine-specific dependency links. The Mac run remains independent.

## Focused follow-up after the main-profile report

`feature-study.mjs` runs a bounded 144-case screen: 16 configurations × brick/UV-grid/wood × three seeds, with 4,000 iterations, batch 4,096, and 1024² sources. It randomizes within seed blocks and does not select or stop variants based on observed quality. Checkerboard is withheld pending validation of its constant baked normal. Road, rotate/scale, and velvet are reserved for follow-up confirmation of finalists.

```sh
# Requires a Python interpreter with matplotlib; use its absolute path here.
NTC_ANALYSIS_PYTHON=/path/to/python node benchmarks/ntc/feature-study.mjs feature-contributions-4000-v2
```

The same command resumes the immutable training snapshot. `findings.md`, `feature-effects.json`, and `feature-effects.png` update every three successful cases and on completion. The ordinary dashboard updates after every case. A single final retry pass handles failed cases. On completion the wrapper also copies the findings and figures into `docs/`. SIGTERM to the wrapper stops after the current case; resume with the same label.

The screen adds two targeted interactions: PE × G0 sampling at uint4, and activation × hidden width. These are conditional interactions; it does not claim the full 18-way encoding/sampling/precision crossing is complete. Mean paired PSNR deltas, mip-0/equal-mip deltas, property deltas, confidence intervals, actual Brotli-size ratios, and estimated grid-memory ratios are reported. The primary reference contrasts additionally use Holm correction within each material, counting planned but missing contrasts in the family. Three seeds provide limited power: failure to detect a difference is not equivalence. Frontier membership uses point estimates from complete seed sets, not statistical dominance.

Statistical reporting checks (run using the analysis Python):

```sh
python benchmarks/ntc/feature_findings_test.py
```
