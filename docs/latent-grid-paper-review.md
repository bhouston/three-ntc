Latent-grid implementation review
================================

Reviewed commit: `402a532`, with a clean working tree before this review. Source: `docs/ntc_small_size.pdf`, especially Sections 4.1–4.4, Figures 4–5, and Tables 1–2. This review covers the representation, sampling, latent training, and storage paths; it does not establish reproduction of the paper's image-quality results.

Concurrent workspace changes appeared at the end of the review in `NTCProfiles.ts`, its test, and the website trainer. The comparison table below records the initial commit's defaults. The new website defaults select `mobile-fast` with G0 widths 128/32, G1 widths 64/16, two 16-neuron hidden layers, and uint2 quantization. At a 1024 source resolution, those two pairs cover mips 0–4 and 5–10. Both grid/encoding flags remain enabled and the eight-texel period remains unspecified. These edits do not resolve the quantization or encoding findings below. They were inspected but were not made by this review.

**Assessment.** The repository implements the paper's central *pyramid of independently learned G0/G1 pairs* when configured appropriately. Its current level selection correctly reproduces Table 1. However, its defaults and bundled examples are materially different, the website uses a different positional-encoding coordinate system, and the four-tap G0 training path has a confirmed quantization-method discrepancy. Calling the entire repository a faithful reproduction without specifying the configuration would be misleading.

**What the paper does.** There are two distinct notions of resolution:

- Within feature level F_j, G0_j is the higher-resolution grid and G1_j is the lower-resolution grid. G1 has half G0's width and height in Table 1. Both contain learned feature vectors; they have different roles and may have different channel counts and quantization depths.
- Across feature levels, each successive pair is spatially coarser. In Table 1, dimensions decrease by four between pairs. Each pair reconstructs several output texture mips.

For the paper's 1024 × 1024 example:

| Feature level | G0 dimensions | G1 dimensions | Reconstructed texture mips |
| --- | --- | --- | --- |
| F0 | 256 × 256 | 128 × 128 | 0, 1, 2, 3 |
| F1 | 64 × 64 | 32 × 32 | 4, 5 |
| F2 | 16 × 16 | 8 × 8 | 6, 7 |
| F3 | 4 × 4 | 2 × 2 | 8, 9, 10 |

For each output texel, the method selects **one feature level** by target LOD, concatenates four neighboring G0 vectors without bilinear averaging, bilinearly interpolates four G1 vectors into one vector, and adds 12 positional-encoding scalars plus normalized LOD. A shared MLP decodes that input. Its input width is `4*C0 + C1 + 12 + 1`, independent of the number of feature levels. Section 4.4 uses two hidden layers of 64 neurons with hardGELU and a linear output.

The G0 inputs support learned interpolation and fine detail; G1 supplies smoothly interpolated content and helps suppress quantization banding. The positional encoding uses triangular waves with three octaves and repeats over an 8 × 8 texel tile (Section 4.3.2/Figure 5).

This is not an encoding that concatenates every spatial resolution into every decoder query. Nor does “pyramid” mean generating coarse learned features by averaging finer learned features. The feature pyramid and decoder are jointly optimized. Sharing each pair across multiple output mips gives the approximately 6.7% additional pyramid storage cited by the paper: successive pair sizes decrease by a factor of 16 in area.

**What the current code implements correctly.**

| Mechanism | Implementation | Assessment |
| --- | --- | --- |
| Separate learned G0/G1 at every level | `createNTCGridPyramidModel()` allocates `grids` and independent `lowResGrids`; G1 dimensions are `floor(G0/2)` | Matches when `dualGrid:true` |
| Sparse feature pyramid | `computeGridLevels()` divides dimensions by `2**mipsPerLevel`; default step is 2 | Matches the paper's factor-of-four spacing |
| One selected pair per query | CPU/TSL level selectors use `floor((lod-lodOffset)/mipsPerLevel)`, clamped to available levels | Matches Table 1 with appropriate source/base resolutions |
| Wider first mip band | `lodOffset=floor(log2(textureResolution/baseResolution))` | For 1024/256, offset 2 gives exactly Table 1 |
| Four raw G0 taps and bilinear G1 | Training and runtime have both paths | Matches when both `positionalEncoding` and `dualGrid` are true |
| Independent gradients | G0 gradients scatter to individual taps; G1 gradients scatter with bilinear weights into the corresponding separate grid | Matches the representation |
| Shared decoder and explicit LOD | Fixed-width feature input and normalized target LOD | Matches |
| Export/import of paired grids | Manifest stores `levels` and `lowResLevels`, plus LOD and encoding metadata | Preserves the current paired representation |

Code anchors: [grid allocation](../packages/three-ntc-trainer/src/NTCGridPyramidModel.ts), [grid resolutions](../packages/three-ntc-trainer/src/NTCGridModel.ts), [LOD selector](../packages/three-ntc/src/NTCMipBands.ts), [training forward/backward paths](../packages/three-ntc-trainer/src/NTCGPUComputeTSL.ts), [runtime sampler](../packages/three-ntc/src/NTCDecoderTSL.ts), [manifest export](../packages/three-ntc-trainer/src/NTCManifest.ts).

The runtime's active path builds native per-grid textures with no generated mipmaps. `evaluateNeuralTextureRaw()` explicitly ignores its legacy `mipChainTexture` argument. Although `NTCHalfFloatTexture.ts` retains box-filtered latent-mip utilities, `NTCNodeMaterial` uses `buildLevelTextures()`. Those old utilities are not evidence that the current material decodes averaged latent pyramids. Explicit trilinear sampling blends decoded texels; stochastic sampling chooses a physical mip and texel.

**The configuration makes a substantial difference.**

| Property | Library trainer defaults | Website trainer defaults | `getNTCPaperProfile(1024)` |
| --- | --- | --- | --- |
| G0 widths | 128, 32, 8, 2 | 256, 64, 16, 4 | 256, 64, 16, 4 |
| G1 widths | Absent | 128, 32, 8, 2 | 128, 32, 8, 2 |
| G0/G1 channels | 4 / absent | 4 / 4 | 8 / 12 |
| G0 sampling | Bilinear | Four raw taps | Four raw taps |
| Positional encoding | Disabled | Grid-cell phase | Eight-texel tile |
| Decoder input width | 5 | 33 | 57 |
| Hidden layers | 32, 32; ReLU | 32, 32; ReLU | 64, 64; hardGELU |
| Latent quantization requested | None during training; uint8 export | uint8 | uint4 for both grids |

At the reviewed commit, the website defaults to a 1024 bake and `mobile-balanced` (see concurrent-change note above). The direct model factory has a different base-resolution fallback from `NTCTrainer`: the factory uses source resolution, capped at 4096, when a base is omitted, while the trainer pre-populates base resolution 128. Neither fallback automatically chooses the paper's quarter-resolution grid. Table 2 also includes half-resolution G0 profiles, so a quarter-resolution base is not universal to every paper configuration.

Sources: [trainer defaults, line 49](../packages/three-ntc-trainer/src/NTCTrainer.ts), [factory defaults, line 83](../packages/three-ntc-trainer/src/NTCGridPyramidModel.ts), [website defaults and trainer construction, lines 89 and 388](../packages/website/src/routes/trainer.tsx), [profiles, line 119](../packages/three-ntc-trainer/src/NTCProfiles.ts).

**Finding 1: G0 ignores noise QAT in the four-tap path — confirmed implementation discrepancy.** Section 4.2 and Figure 4 apply simulated quantization noise to both grids before sampling/concatenation, then quantize and freeze their values for final decoder adaptation.

In `NTCGPUComputeTSL.ts`, `readTap()` at line 222 handles noise versus STE and is used by bilinear sampling, including G1. But the four-tap G0 branch at lines 246–256 directly invokes `quantizeForwardTSL`, bypassing `readTap()`. Consequently, with `method:'noise'`, G0 is hard-rounded while G1 receives noise. The paper profile triggers this path.

A temporary GPU probe used constant latent values 0.02, uint4 noise QAT, and both grid/encoding flags enabled. All four G0 inputs became exactly zero, whether the noise uniform was enabled or disabled. The G1 input varied with noise and returned to approximately 0.02 when noise was disabled. This confirms the discrepancy beyond static inspection. The existing noise test covers the bilinear G0 configuration, so it does not catch it.

Recommended correction: use the same `readTap()` implementation for the four G0 taps and add coverage with both `positionalEncoding:true` and `dualGrid:true`. Its impact on final reconstruction quality needs an experiment; the method mismatch itself is established.

**Finding 2: enabling positional encoding does not select the paper's tile coordinates.** The model defaults `positionalEncodingPeriod` to zero. That selects the fractional position within the selected G0 cell. Only a nonzero period selects target-mip texel coordinates; the paper helper supplies 8. The website supplies `positionalEncoding:true` but omits the period.

At the website's default 1024/256 ratio, grid-cell phase repeats every four mip-0 texels, rather than the paper's eight. At other mips its scale changes with the source/grid ratio. Training and decoding agree with each other, but the encoding differs from the paper. See [training encoding, line 280](../packages/three-ntc-trainer/src/NTCGPUComputeTSL.ts), [runtime encoding, line 76](../packages/three-ntc/src/NTCDecoderTSL.ts), and [triangle-wave implementation](../packages/three-ntc/src/NTCPositionalEncodingTSL.ts).

Recommended correction: explicitly set period 8 for new paper-style training, while retaining compatibility for old assets. The passing encoding GPU test establishes the eight-texel repetition property; it does not independently establish exact numerical equivalence to every scalar depicted in Figure 5.

**Finding 3: the paper helper does not reproduce a Table 2 bitrate profile.** Its 8/12 channels match the paper's NTC 0.2 channel counts, but Table 2 uses G0 at 2 bits and G1 at 4 bits. The helper uses 4 bits for both. The other Table 2 profiles are 12/20 channels at 4/4 bits, 12/10 at 2/4 bits, and 16/12 at 4/4 bits, with the finest grid resolutions shown there.

The training configuration has one quantization mode, and `encodeNTC()` chooses one dtype for all G0/G1 grids. Although individual manifest grids carry a dtype, the current training/export interface does not implement those mixed 2/4-bit profiles. For the helper's grid shape, latent storage is `44/28`, approximately 1.57 times the NTC 0.2 latent payload: `8*4 + 12*4/4` versus `8*2 + 12*4/4` bits per G0 cell, before network/format overhead. The helper's comment acknowledges the shared uint4 choice.

Recommended correction: separate G0/G1 bit-depth configuration if reproducing paper rate-distortion results is the objective.

**Finding 4: the eight bundled examples use the older single-grid model.** All files in `packages/website/public/ntc/` contain three G0 grids with widths 128, 32, 8, four channels, and uint8 storage. None contains G1 grids, enabled positional encoding, or the new LOD-offset metadata. The loader defaults missing offset to zero, preserving the old band mapping rather than recomputing Table 1 behavior.

These files are the website's listed demo assets. Their appearance therefore does not evaluate the newly supported G0/G1 architecture. They need retraining to evaluate that architecture; changing their flags would not make their trained decoder weights compatible.

**Additional distinctions and limitations.**

- The generic grid builder accepts arbitrary level counts and continues appending 1 × 1 grids after reaching the minimum. It does not automatically stop at the paper's final 4 × 4 / 2 × 2 pair. The 1024 paper helper does stop at that pair; arbitrary user settings need separate validation.
- Both training and runtime express selection by sampling all stored levels and multiplying unselected values by zero. Mathematically this selects one pair, so it is not multilevel concatenation. It may nevertheless issue unnecessary texture/buffer reads. Actual compiler elimination and cost were not profiled here.
- Runtime storage expands quantized features to RGBA16F array textures, with four-channel packing. This differs from keeping the low-bit latent representation resident for decoding and means file payload size does not equal GPU latent memory consumption. See `buildLevelTextures()` at line 96 of `NTCHalfFloatTexture.ts`.
- Runtime retains a fallback that uses the coarsest G0 as G1 when a dual-grid asset lacks separate low-resolution grids. That is not the paper's independent pair representation. Newly allocated/exported dual-grid models do have separate G1 grids and do not use the fallback.
- Comments are inconsistent with current code: `NTCGridModel.ts` still says the first mip band is uniformly sized, and the training scatter comment around line 434 still describes G1 as an unconditional coarsest-grid read. The executable code now has the source/base offset and independently allocated, gated G1 gradients. Those comments should be corrected.

**Validation and proposed order of work.** Five focused unit-test files passed, comprising 15 tests covering grid construction, paired export/load, profiles, quantization, and Table 1 selection. Three GPU test files passed, comprising four tests: the existing encoding/noise tests plus the temporary discrepancy probe. The probe was removed afterward; no implementation files were changed. No full training run or paper-dataset quality comparison was performed.

First correct the G0 noise path and explicitly select eight-texel encoding in paper-style callers. Then add separate grid bit depths and an exact Table 2 preset, retrain a representative demo, and compare reconstructed mips and storage at the same configuration. Preserve the existing correct paired-pyramid design and LOD mapping. Update comments and distinguish custom mobile presets from paper-reproduction presets so that future comparisons use the intended architecture.
