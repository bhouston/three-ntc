import { describe, expect, it } from 'vitest';

import { NTCLoader } from './NTCLoader.js';
import { FORMAT, VERSION } from './NTCFormat.js';
import { encodeUint8Base64, encodeMLPLayersBase64 } from './NTCBinaryCodec.js';

// Builds the smallest manifest NTCLoader.parse() accepts: one 2x2 grid
// level, a single-layer linear decoder, one active channel - just enough to
// exercise the `latents.positionalEncoding` round-trip (see
// NTCGridPyramidModel.js/NTCDecoderTSL.js, three-ntc-trainer).
function buildManifest(positionalEncoding?: boolean, dualGrid?: boolean) {
  const channels = 1;
  const inputSize = (positionalEncoding ? 4 * channels + 12 : channels) + (dualGrid ? channels : 0) + 1;

  return {
    format: FORMAT,
    version: VERSION,
    latents: {
      channelsPerLevel: channels,
      mipsPerLevel: 2,
      maxLod: 1,
      levels: [
        {
          width: 2,
          height: 2,
          channels,
          dtype: 'uint8' as const,
          min: 0,
          max: 1,
          dataBase64: encodeUint8Base64(new Float32Array([0, 0.25, 0.5, 0.75]), 0, 1),
        },
      ],
      ...(positionalEncoding !== undefined ? { positionalEncoding } : {}),
      ...(dualGrid !== undefined ? { dualGrid } : {}),
    },
    mlp: encodeMLPLayersBase64([
      { inputSize, outputSize: 1, activation: 'linear', weights: new Float32Array(inputSize).fill(0), biases: new Float32Array(1) },
    ]),
    channels: { activeKeys: ['roughness'] },
  };
}

describe('NTCLoader positionalEncoding flag', () => {
  it('defaults to false when the manifest predates the field', () => {
    const { cpuModel } = new NTCLoader().parse(buildManifest(undefined) as any);
    expect(cpuModel.positionalEncoding).toBe(false);
    expect(cpuModel.dualGrid).toBe(false);
  });

  it('round-trips dualGrid, sized for the extra G1 tap', () => {
    const { cpuModel } = new NTCLoader().parse(buildManifest(false, true) as any);
    expect(cpuModel.dualGrid).toBe(true);
    expect(cpuModel.decoder.layers[0].inputSize).toBe(1 + 1 + 1);
  });

  it('round-trips an explicit true, sized for the 4-tap + posenc decoder input', () => {
    const { cpuModel } = new NTCLoader().parse(buildManifest(true) as any);
    expect(cpuModel.positionalEncoding).toBe(true);
    expect(cpuModel.decoder.layers[0].inputSize).toBe(4 * 1 + 12 + 1);
  });
});

it('preserves legacy emissive sigmoid and reads explicit HDR activations', () => {
  const manifest = buildManifest() as any;
  manifest.channels.activeKeys = ['emissive'];
  const loader = new NTCLoader();
  expect(loader.parse(manifest).channelClassification.activeChannels[0].activation).toBe('sigmoid');
  manifest.channels.encodings = {emissive:{activation:'softplus'}};
  expect(loader.parse(manifest).channelClassification.activeChannels[0].activation).toBe('softplus');
  manifest.channels.encodings.emissive.activation = 'unknown';
  expect(()=>loader.parse(manifest)).toThrow(/Invalid activation/);
});
