---
title: How neural texture compression works
sidebar_position: 4
description: Feature grids, neural decoding, mipmaps, quantization, and the implementation's tradeoffs.
---

A PBR material often stores multiple images covering the same UV domain: base color, roughness, metalness, normals, and other properties. These images share structure. A scratch may affect both color and roughness; a brick boundary can appear in several channels. Neural texture compression learns a compact representation of this correlated data.

The research foundation is NVIDIA's SIGGRAPH 2023 paper, [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/), by Vaidyanathan and colleagues. Its central idea is joint compression of material textures and their mip chains, with a small network enabling random-access decoding. The project page links the paper, supplementary material, video, and comparisons.

three-ntc implements this family of ideas with three.js TSL and WebGPU. Its defaults, browser constraints, file format, and memory layout are implementation choices. Results in the paper are not benchmark claims for this library, and this project's JSON `.ntc` format should not be assumed compatible with NVIDIA tools.

## A learned representation of one material

The representation has two trainable parts:

1. **Feature grids:** spatial arrays of short numerical vectors, organized at several resolutions. These vectors are learned latent features, not RGB pixels.
2. **A decoder:** a small multilayer perceptron (MLP) shared across the material. Given grid features and sampling information, it outputs the material's active channels.

At a high level, a query follows this path:

```text
mesh UV + screen-space footprint
          |
          v
select physical mip and corresponding feature-grid band
          |
          v
read learned features (+ optional positional information)
          |
          v
shared MLP -> channel activations -> physical material shading
```

The network is optimized for this material. Loading an asset loads both its features and its decoder; it does not call a remote AI model or generate a new texture from a prompt.

## Why share the decoder?

Rather than training an independent texture compressor for each map, the decoder reconstructs the active material channels together. This gives the representation a chance to reuse spatial correlations and amortizes the decoding work across properties.

The trainer classifies source channels before fitting. Spatially varying channels are packed into the learned output layout. Constant values are stored separately and applied directly at runtime, so the network does not spend capacity learning a flat number.

Channel-specific activations map raw network output into useful ranges: bounded channels can use sigmoid, signed values can use tanh, and HDR emission can use a nonnegative activation without forcing it into the unit interval. The manifest records channel encoding choices needed for reconstruction.

## Grid capacity and decoder capacity

A denser grid stores more local information. A wider or deeper decoder can fit more complex relationships, at increased ALU cost every time a sample is decoded. A representation with too little capacity tends to smooth or lose detail; more iterations cannot always overcome that bottleneck.

The default browser trainer uses a single grid pyramid, four feature channels, two 32-wide ReLU hidden layers, and no positional encoding. Optional `dualGrid` adds a second, lower-resolution grid stream; optional positional encoding supplies richer local position information to the decoder. The paper-oriented preset uses both, with a 64-wide hardGELU decoder. These options increase the input size and decode cost as well as representational capacity.

The named mobile and desktop profiles are practical architecture presets. They do not reproduce the paper's named bitrate targets. Actual bits per texel depend on source resolution, active channel count, grid dimensions, precision, and metadata.

## Mips are learned too

Ordinary textures use mipmaps to avoid aliasing when a pixel covers many texels. A neural representation needs the same concept: querying the finest detail at every distance is not sufficient.

The trainer samples across the physical mip range. Several physical mip levels share each stored feature-grid level, with LOD information telling the decoder what scale to reconstruct. The runtime estimates a continuous LOD from screen-space derivatives of the UV coordinates, using the original source resolution. Derivatives are taken before repeat wrapping so tile seams do not create artificial LOD spikes.

Runtime filtering operates on the reconstructed **physical texels**, separately from the feature interpolation inside the model. Nearest reconstruction costs one decoder evaluation. Deterministic trilinear reconstruction evaluates eight physical samples. Stochastic filtering chooses one of those samples probabilistically, reducing work at the cost of noise. See [runtime integration](./runtime.md) for controls and the temporal-reconstruction caveat.

## Training minimizes reconstruction error

Training bakes source material channels to GPU textures and repeatedly samples UV/mip locations. The decoder predicts those samples, an L2 reconstruction loss measures error, and Adam updates both the feature grids and decoder weights. Learning rates decay over the requested budget. Most work stays on the GPU, with periodic synchronization for progress previews and metrics.

The loss measures error in the training channels, not perceptual equivalence under every light and viewpoint. Tiny normal or roughness errors can noticeably change a highlight. Always compare shaded materials as well as channel views, including close views, grazing angles, minification, and the final exported asset.

## Quantization and the export boundary

Storing every learned feature as a full float would waste much of the compression opportunity. Quantization maps feature values to a finite set of levels. Lower precision saves bytes but introduces error.

Quantization-aware training exposes the optimizer to this error before export. This implementation supports latent `uint8`, `uint4`, and `uint2` codecs. With quantization enabled, the final retraining phase freezes rounded latents and lets the decoder adapt. Decoder weights are float16-packed on disk.

The asset is a JSON manifest containing metadata and base64-packed arrays. At load time those arrays expand into CPU data, GPU feature textures, and decoder parameters. Exported bytes and resident GPU bytes are therefore different quantities; measure both for your application.

## When to use it

The trade is storage and bandwidth against reconstruction error, training time, and runtime compute. It can be useful when a material has many correlated channels and a compact download is valuable. It is less attractive when an ordinary constant material suffices, when exact texel reproduction is required, or when decode cost dominates the scene.

Evaluate on your actual target hardware. Compare total download size, resident memory, frame time, and visual quality with conventional texture formats using equivalent source resolution, mip coverage, and rendering settings. This library uses WebGPU shader computation, so paper benchmarks from a different implementation cannot predict its frame time.
