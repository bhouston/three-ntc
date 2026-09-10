# Browser performance profiles

Run from the repository root, with no other performance test running:

```sh
pnpm exec playwright install chromium webkit
pnpm profile:browser chromium /tmp/ntc-chromium
pnpm profile:browser webkit /tmp/ntc-webkit
pnpm profile:browser safari /tmp/ntc-safari
```

The runner executes each browser's workloads serially and saves `results.json`
plus complete logs, including launch failures. Chromium and Playwright WebKit run
at DPR 1 and 2. Native Safari uses the real display's DPR, which the viewer metrics
record. On macOS the runner defaults Chromium to Metal; set `NTC_GPU_BACKEND` to
choose another backend. Adapter information is recorded so software rendering
cannot be silently compared with hardware. Chromium is the automated Chrome-engine
target; it is not the user's installed Google Chrome application.

Safari uses Apple's `safaridriver` through Vitest's WebdriverIO provider. Remote
automation must be enabled in Safari. A successful WebKit run does not establish
that installed Safari passes: its version, driver, and browser configuration differ.
The September 10 retry still failed during native session creation, before any test
executed. The runner reports this as a failure, not a skipped or passing profile.

## Workloads

- **Actual website viewers:** standalone brick, trainer comparison, and six-sphere
  brick gallery, each at 512×384 and 1024×768 CSS pixels. Each loads the actual HDR
  environment and completes 120 animation frames. The first 20 frames are excluded
  from steady-state timing summaries. At DPR 2 the larger canvas is 2048×1536.
  These six cases default to nearest sampling. Two additional 1024px standalone
  cases exercise stochastic and trilinear, with the mode recorded in each metric.
- **Training with an animated preview:** five default-size training steps, two
  model updates, and a concurrently rendering physical preview on a separate GPU
  device. Records loss, update costs, animation intervals, browser heartbeat stalls,
  and shader-module counts. The source is a synthetic 1024px brick; the preview is
  a 256px offscreen plane, so it does not reproduce the complete website trainer.
- **Shipped-brick decoder check:** physical rendering and GPU-versus-CPU decoder
  error, to retain a numerical quality check alongside the performance measurements.

Viewer metrics retain raw frame intervals, submit-to-GPU-completion delays, and
heartbeat gaps. Summaries report count, mean, p50, p95, p99, and maximum; percentiles
use nearest rank. First-frame time includes asset loading and pipeline work. Timer
heartbeats cover startup too; they measure event-loop responsiveness, not input
latency. Completion delays include queued GPU work and are not GPU timestamp-query
execution durations. Frames use the browser animation loop without awaiting each
GPU completion, which lets these tests expose accumulating GPU work.

Tests require completed frames, no observed WebGPU errors, and disabled MSAA.
Training also checks finite output/loss and no shader recompilation after the first
preview. Timing values are observations, not arbitrary pass/fail FPS thresholds.
Compare repeated runs on the same machine, resolution, backend, browser version,
and power settings; inspect tail stalls as well as medians. These roughly two-second
viewer cases detect sustained regressions but are not long-duration soak tests.

For a quick targeted run:

```sh
NTC_BROWSER=webkit NTC_DEVICE_SCALE_FACTOR=2 pnpm test:gpu --fileParallelism=false packages/website/src/components/NTCViewer.gpu.test.ts
pnpm test:gpu:safari packages/website/src/components/NTCViewer.gpu.test.ts
```

See [the cause summary](viewer_performance_summary.md) and
[the investigation](../tsl_performance_regression_cause_and_fix.md) for the MSAA
before/after measurements. The larger matrix introduced afterward should establish
new baselines; it should not be treated as directly identical to the old 25-frame runs.

The initial expanded matrix's [summary baseline](metrics/browser-profile-baseline.json)
records 16 passing cases per engine for Chromium and WebKit, plus the failed native
Safari launch. It also retains the slower initial training profiles: startup heartbeat
stalls reached 379 ms in Chromium and 512 ms in WebKit even though steady viewer
frame intervals were near 17 ms. Cache state affects repeat runs; these startup
pauses remain useful targets for further investigation.
