# Why the neural viewers were slow

Runtime sampling is now configurable: **nearest (default)** and **stochastic**
evaluate the decoder once; optional **trilinear** evaluates it eight times. The
paper describes all three, and uses stochastic filtering with temporal reconstruction
for its main rendering results. We have not added temporal reconstruction here.
The eight-decode implementation below was an expensive choice, not a requirement
for every NTC sample. See [runtime sampling](../packages/three-ntc/README.md#runtime-sampling).

Two separate problems appeared during this investigation:

1. **Shader expansion and repeated compilation.** The earlier runtime
   used explicit trilinear filtering, evaluating the neural decoder eight times. Expanding those calls and the MLP
   arithmetic into a large shader caused excessive private storage and expensive
   compilation. Runtime loops for filtering and larger matrix layers reduced the
   shader size. Reusing the preview material during training also avoids compiling
   a new material after each progress update. Materializing intermediate TSL values
   with `toVar()` helps avoid repeated expressions, but did not solve everything.
2. **MSAA in WebKit at Retina resolution.** After those fixes, the actual website
   viewer still had sustained poor performance. Offscreen tests missed it because
   they did not use the website's 4× MSAA. With the shipped brick and HDR lighting
   on a 2048×1536 canvas, Playwright WebKit produced 50–60 ms frame intervals and
   650–657 ms submit-to-completion delays near the end of the measured run.
   Chromium/Metal stayed near 60 fps with MSAA. Disabling MSAA in the same WebKit
   workload reduced the intervals to 16–17 ms and completion delays to 10–11 ms.

The long completion delays include queued GPU work, not just one frame's execution.
We isolated MSAA as a major performance factor in this workload; we have **not**
established the exact WebKit/Metal compiler or driver mechanism behind it.

All website renderers now explicitly disable MSAA, including the gallery and the
trainer's comparison preview. FXAA is not enabled. Canvas resolution and the neural
model are unchanged. The tradeoff is less smoothing of geometry edges.

Actual-component tests cover the standalone brick, trainer comparison, and gallery.
All three passed in Chromium and Playwright WebKit at DPR 2, with approximately
17 ms median frame intervals in WebKit and no WebGPU errors. These short profiles
are evidence for the tested workload, not a guarantee for every material or device.
Installed Safari is a separate target: its native driver accepted the enabled
remote-automation setting but timed out creating a session, so these results must
not be presented as measurements from installed Safari.

For repeatable Chrome-engine and Safari/WebKit workloads, see
[browser performance tests](browser_performance_tests.md).

See [the full investigation](../tsl_performance_regression_cause_and_fix.md),
[raw before/after measurements](metrics/runtime-msaa.json), and
[the progress report](../astra_progress.md).
