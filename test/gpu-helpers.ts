// Shared helpers for the `*.gpu.test.ts` browser/WebGPU tests (see the
// `gpu` project in /vitest.config.ts): one lazily-initialized renderer, a
// 'render this TSL vec4 at every UV and read the floats back' harness, and
// a plain-JS CPU reference decoder that mirrors NTCDecoderTSL.ts /
// NTCGPUComputeTSL.ts. The CPU decoder is the oracle the GPU paths are
// checked against - it is deliberately written independently of the TSL
// code (plain loops, no shared helpers beyond the tiny pure functions the
// packages already export) so the two can disagree.
import { DataUtils } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { selectFeatureLevel, POSITIONAL_ENCODING_OCTAVES } from 'three-ntc';
import { bakeColorNodeToTexture, createNTCGridPyramidModel, createRandom } from 'three-ntc-trainer';

let rendererInit: Promise<any> | null = null;

export function getRenderer(): Promise<any> {
  rendererInit ??= (async () => {
    const renderer = new WebGPURenderer({ antialias: false });
    await renderer.init();
    return renderer;
  })();
  return rendererInit;
}

/**
 * Renders `vec4Node` across a `size` x `size` quad and returns RGBA floats,
 * row-major. Pixel (x, y) saw `uv() === pixelUv(size, x, y)` - verified by
 * the harness sanity test in NTCDecoder.gpu.test.ts.
 */
export async function renderNodeToFloats(
  renderer: any,
  vec4Node: any,
  size: number,
): Promise<Float32Array> {
  const renderTarget = await bakeColorNodeToTexture(renderer, vec4Node, size);
  const out = await readRenderTargetFloats(renderer, renderTarget, size);
  renderTarget.dispose();
  return out;
}

/** Reads a `size` x `size` half-float render target back as RGBA floats. */
export async function readRenderTargetFloats(
  renderer: any,
  renderTarget: any,
  size: number,
): Promise<Float32Array> {
  const half: Uint16Array = await renderer.readRenderTargetPixelsAsync(
    renderTarget,
    0,
    0,
    size,
    size,
  );
  // WebGPU buffer readback rows are padded to 256 bytes (128 half-floats);
  // three.js hands that padding back for widths under 32 texels.
  const stride = Math.max(size * 4, 128);
  const out = new Float32Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let i = 0; i < size * 4; i++)
      out[y * size * 4 + i] = DataUtils.fromHalfFloat(half[y * stride + i]);
  }
  return out;
}

export function pixelUv(size: number, x: number, y: number): [number, number] {
  return [(x + 0.5) / size, (y + 0.5) / size];
}

export function roundHalf(x: number): number {
  return DataUtils.fromHalfFloat(DataUtils.toHalfFloat(x));
}

export function wrap(i: number, n: number): number {
  return ((i % n) + n) % n;
}

/** Wrap-addressed bilinear sample at texel-center convention (x = u*w - 0.5). */
export function bilinearWrap(
  data: ArrayLike<number>,
  width: number,
  height: number,
  channels: number,
  u: number,
  v: number,
  texel: (x: number) => number = (x) => x,
): number[] {
  const x = u * width - 0.5;
  const y = v * height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const out: number[] = Array.from({ length: channels }, () => 0);
  const taps: Array<[number, number, number]> = [
    [x0, y0, (1 - tx) * (1 - ty)],
    [x0 + 1, y0, tx * (1 - ty)],
    [x0, y0 + 1, (1 - tx) * ty],
    [x0 + 1, y0 + 1, tx * ty],
  ];
  for (const [tx_, ty_, w] of taps) {
    const base = (wrap(ty_, height) * width + wrap(tx_, width)) * channels;
    for (let c = 0; c < channels; c++) out[c] += texel(data[base + c]) * w;
  }
  return out;
}

function triangleWave(x: number): number {
  return Math.abs(x - Math.floor(x) - 0.5) * 4 - 1;
}

export function positionalEncoding(tx: number, ty: number): number[] {
  const values: number[] = [];
  for (const t of [tx, ty]) {
    for (let h = 0; h < POSITIONAL_ENCODING_OCTAVES; h++) {
      const freq = 2 ** h;
      values.push(triangleWave(t * freq - 0.25));
      values.push(triangleWave(t * freq));
    }
  }
  return values;
}

export function activate(z: number, activation: string): number {
  if (activation === 'relu') return Math.max(0, z);
  if (activation === 'hgelu') return z <= -1.5 ? 0 : z >= 1.5 ? z : (z / 3) * (z + 1.5);
  return z;
}

export function activateDerivative(z: number, activation: string): number {
  if (activation === 'relu') return z > 0 ? 1 : 0;
  if (activation === 'hgelu') return z <= -1.5 ? 0 : z >= 1.5 ? 1 : (2 * z + 1.5) / 3;
  return 1;
}

export function channelActivate(z: number, activation?: string): number {
  if (activation === 'sigmoid') return 1 / (1 + Math.exp(-z));
  if (activation === 'tanh') return Math.tanh(z);
  if (activation === 'softplus') return Math.max(z, 0) + Math.log(1 + Math.exp(-Math.abs(z)));
  return z;
}

export function channelActivateDerivativeFromOutput(a: number, activation?: string): number {
  if (activation === 'sigmoid') return a * (1 - a);
  if (activation === 'tanh') return 1 - a * a;
  if (activation === 'softplus') return 1 - Math.exp(-a);
  return 1;
}

/** Forward pass returning every layer's pre-activation `z` and activation `a` (a[0] = input). */
export function forwardCpu(
  decoder: { layers: any[] },
  input: number[],
): { z: number[][]; a: number[][] } {
  const z: number[][] = [];
  const a: number[][] = [input.slice()];
  for (const layer of decoder.layers) {
    const zl: number[] = [];
    const al: number[] = [];
    for (let o = 0; o < layer.outputSize; o++) {
      let s = layer.biases[o];
      for (let i = 0; i < layer.inputSize; i++)
        s += layer.weights[o * layer.inputSize + i] * a[a.length - 1][i];
      zl.push(s);
      al.push(activate(s, layer.activation));
    }
    z.push(zl);
    a.push(al);
  }
  return { z, a };
}

/**
 * The decoder's input feature vector as the *runtime* decoder builds it
 * (hardware trilinear over the half-float mip chain for the plain path,
 * nearest 4-tap + positional encoding otherwise), followed by the LOD scalar.
 */
export function runtimeFeaturesCpu(cpuModel: any, u: number, v: number, lod: number): number[] {
  const { channels, grids, mipsPerLevel, maxLod } = cpuModel;
  let features: number[];
  if (cpuModel.positionalEncoding) {
    const g = selectFeatureLevel(lod, grids.length, mipsPerLevel, cpuModel.lodOffset);
    const grid = grids[g];
    const x = u * grid.width - 0.5;
    const y = v * grid.height - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    features = [];
    for (const [dx, dy] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]) {
      const base = (wrap(y0 + dy, grid.height) * grid.width + wrap(x0 + dx, grid.width)) * channels;
      for (let c = 0; c < channels; c++) features.push(roundHalf(grid.data[base + c]));
    }
    features.push(...positionalEncoding(x - x0, y - y0));
  } else {
    const grid = grids[selectFeatureLevel(lod, grids.length, mipsPerLevel, cpuModel.lodOffset)];
    features = bilinearWrap(grid.data, grid.width, grid.height, channels, u, v, roundHalf);
  }
  if (cpuModel.dualGrid) {
    const last = cpuModel.lowResGrids?.[selectFeatureLevel(lod,grids.length,mipsPerLevel,cpuModel.lodOffset)] ?? grids[grids.length - 1];
    features.push(...bilinearWrap(last.data, last.width, last.height, last.channels, u, v, roundHalf));
  }
  features.push(lod / Math.max(1, maxLod));
  return features;
}

/** Raw decoder outputs (before any per-channel output activation) at (u, v, lod). */
export function evaluateNTCCpu(cpuModel: any, u: number, v: number, lod: number): number[] {
  const { a } = forwardCpu(cpuModel.decoder, runtimeFeaturesCpu(cpuModel, u, v, lod));
  return a[a.length - 1];
}

export function makeModel(seed: number, options: Record<string, unknown>): any {
  return createNTCGridPyramidModel(options as any, createRandom(seed));
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Peak signal-to-noise ratio (dB) over the first `channels` of every RGBA pixel. */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>, channels = 3): number {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < channels; c++) {
      const d = a[i + c] - b[i + c];
      sum += d * d;
      n++;
    }
  }
  return 10 * Math.log10(1 / (sum / n));
}
