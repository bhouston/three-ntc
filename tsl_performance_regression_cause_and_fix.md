# TSL runtime performance regression: cause, experiments, and fix

This investigation concerns the trainer preview freezing and eventually reporting
WebGPU device loss. The initial private-storage validation error and the subsequent
severe slowdown are distinct symptoms. Fixing the validation error does **not**
establish that the runtime is usable.

## Evidence in the original repository

The available checkout is `../three.js-v2-basic-ntc` (the requested `v3` directory
is absent). Its history contains commit
`9e40819833a15fbb90450a206819846131469598`, dated 2026-08-20, titled
**“fix neural-appearance performance issue finally.”** Inspect it with:

```sh
git -C ../three.js-v2-basic-ntc show 9e40819833
```

The deleted `examples/jsm/neural-appearance/NeuralAppearanceTSL.js` contains
`evaluateMLPViaFn`. Its comments explain two different structural safeguards:

1. Materialize each layer's outputs with `.toVar()` so downstream consumers do
   not repeatedly expand the preceding expression graph.
2. Wrap the MLP in `Fn(...).setLayout(...)` to emit a real shader function with
   local temporaries. An inline `Fn(...)()` alone does not supply this function
   boundary. Multiple inline networks can accumulate enough module-level
   `var<private>` storage to exceed the reported 8,192-byte limit.

The surviving `examples/jsm/ntc/NTCMLPTSL.js` repeats this warning, and
`NTCDecoderTSL.js` mentions the related “maximum parser recursive depth” failure.
These are source evidence for a known structural failure, not proof that every
current slowdown has the same cause.

## Structural difference introduced here

The original NTC runtime samples an interpolated feature texture and evaluates
**one** MLP per pixel. Our paper-correct reconstruction decodes the four neighboring
texels at each of two mip levels, applies output activations, and then blends:
**eight** evaluations per pixel. Interpolating latent features before a nonlinear
MLP is not equivalent to interpolating its reconstructed outputs.

The first implementation expanded those eight evaluations in JavaScript. Three
then generated separate temporaries for each network. The replacement shader loop
reuses the decoder graph and lowers private storage, but still contains a large
statically expanded matrix computation. This distinction matters:

```ts
// Builds repeated shader code during JavaScript graph construction.
for (let tap = 0; tap < 8; tap++) evaluateDecoder(...);

// Emits a shader loop containing one decoder graph.
Loop(8, ({ i }) => evaluateDecoder(...));
```

The dense layers also build matrix-vector sums in JavaScript. Adding `.toVar()`
at the end of a sum, materializing every addition, and changing that sum into a
shader loop are different structures even when their mathematics agrees.

## Browser measurements so far

Vitest's `gpu` project runs in Playwright Chromium with WebGPU.
The original measurements below used **SwiftShader (software rendering)**, as
confirmed by the adapter information; they must not be presented as Apple GPU
measurements. The test uses the
actual physical NTC material, seven output channels, four feature levels,
positional encoding, a dual grid, and a 32–32 hidden network. It checks decoded
values against an independent CPU decode-then-filter oracle at fractional LOD.
The render target is only 16 × 16: passing this test is not a claim of interactive
performance at viewer resolution or on Safari.

| Arrangement | Test duration | Conservative private bytes | CPU-reference MSE |
| --- | ---: | ---: | ---: |
| Eight-tap shader loop, original matrix sums | 31.61 s | 3,940 | 3.4234e-11 |
| Same, mutable addition accumulators | 30.53 s | 4,228 | 3.4234e-11 |
| Same, real function around MLP | 31.26 s | 3,716 | 3.4234e-11 |

These are individual runs, not a statistically controlled benchmark. None shows
a useful speed improvement. A separately instrumented scoped-function run spent
134 ms in `compileAsync`, **11,113 ms** in the first render plus completed readback,
and **19,263 ms** in the two additional oracle renders. This means timing only
`compileAsync` would badly misrepresent readiness; first-use driver work may be
deferred. The measurements do not isolate GPU execution from driver compilation.

The 64-wide variant has also stalled in experiments. A fix must pass both sizes,
not merely fit below the private-storage budget.

## Preview lifecycle issue

The trainer reports progress at iterations 1, 5, 9, and so on. Previously each
report rebuilt the neural material, replacing its graph, uniforms, and textures.
That can repeat the expensive first-use work at exactly the observed pause cadence.

The pending preview change updates weight uniforms and latent texture data in
place when model structure is unchanged. Its browser regression test checks that
pixels change as expected while texture identities, material version, and shader
module creation count remain unchanged. A new network shape still requires a new
material.

This avoids repeated compilation; it cannot by itself repair an intrinsically
slow shader.

## Confirmed structural improvement: runtime matrix loops

The matrix sum now uses two nested shader loops over output and input vec4
blocks, with loop bounds held in uniforms. This keeps the WGSL dense computation
compact instead of expanding every matrix block in JavaScript. The mat4 × vec4
multiplication itself is retained; this does not change weights or arithmetic order.

```ts
const inputs = array(packedInputs).toVar();
Loop({ end: outputCountUniform, name: 'ntcOutput' }, ({ ntcOutput }) => {
  const sum = biases.element(ntcOutput).toVar();
  Loop({ end: inputCountUniform, name: 'ntcInput' }, ({ ntcInput }) => {
    sum.addAssign(weights.element(ntcOutput.mul(inputCount).add(ntcInput))
      .mul(inputs.element(ntcInput)));
  });
  outputs.element(ntcOutput).assign(activation(sum));
});
```

Both loops must have distinct shader variable names. Nested `Loop(count)` calls
both default to `i`; renaming the JavaScript destructured argument does not rename
the WGSL variable. The CPU oracle caught the resulting incorrect indexing during
development. An inline `Fn` must also enclose the assignments so raw decoder callers
outside an existing TSL function register them correctly.

| Matrix arrangement | 32-wide first render/readback | 64-wide first render/readback |
| --- | ---: | ---: |
| Input-block shader loop only | 1,943 ms | 3,764 ms |
| Both dimensions as shader loops | 663 ms | 736 ms |

The latter's CPU-reference MSE remained **3.4234e-11** (32-wide) and **5.1345e-11**
(64-wide). Private-storage upper bounds were 4,388 and 4,900 bytes. A subsequent
full runtime regression run passed all **35 browser tests**, including shipped
assets, raw decodes, activations, and filtering. Its first-frame measurements
varied to 805 and 1,068 ms, illustrating why timing is reported rather than asserted
against a narrow machine-dependent threshold.

The controlled changes establish that static expansion of the matrix computation
is a major contributor to first-use latency on this test backend. They do not
establish which internal driver optimization caused the stall, nor isolate the
effect of uniform bounds from the compact loop structure.

## Live preview profile and remaining limitation

The new browser profile uses separate viewing/training WebGPU devices, a synthetic
1024 × 1024 brick source, the website's 32–32 network/dual-grid/positional encoding,
8,192 samples per batch, and five training iterations. A 256 × 256 physical preview
is rendered at iterations 1 and 5. Measured elapsed time was 3.658 seconds, with a
62.4 ms maximum 16-ms timer heartbeat gap, two shader modules total, and no WebGPU
errors. Reported training loss moved from 0.29954 to 0.23010; this is a smoke-run
observation, not a comparative convergence benchmark.

**Steady frames still took 214–238 ms. The user likewise reports approximately
6 fps after the changes. The performance problem is not fully resolved.** Avoiding
freezes and repeated shader creation is an improvement, not an interactive-runtime
acceptance result. Timing includes completed GPU work, with shader work and GPU
execution not separated by timestamp queries.

Further investigation must address steady rendering as well as startup latency.
The current sampler fetches all feature levels and multiplies unwanted levels by
zero; skipping those samples is a candidate, not yet a measured fix.

The final regression should record:

- Completed first-frame and subsequent-frame times, not just submission time.
- Browser heartbeat gaps during training and rendering.
- WebGPU validation errors and device loss, with failures surfaced by the test.
- Two training preview updates with no shader recompilation after initialization.
- CPU-reference reconstruction error so a faster but incorrect shader cannot pass.

Timing observations should be retained in `astra_progress.md`; numerical error
and structural invariants should be asserted separately from machine-dependent
performance measurements. A fresh page is needed to retest after device loss:
updating code does not revive an already-lost GPU device.

## Hardware-backend check

`NTC_GPU_BACKEND=metal` selects Chromium's Metal backend on macOS. The same
training-preview profile then reports adapter `apple / metal-3`, completes in
1.066 seconds, and renders steady 256 × 256 frames in 5.5–6.3 ms. This differs
substantially from SwiftShader. A standalone 1024-pixel sphere using the shipped
brick (5–8–8–10 MLP), two directional lights, and ambient light measured 5.2–9.3 ms
steady frames on Metal. That initial standalone test omitted the viewer HDR map;
the matching HDR workload is being tested. The user's approximately 6 fps is still
unresolved and cannot be dismissed based on a smaller or incomplete test scene.
