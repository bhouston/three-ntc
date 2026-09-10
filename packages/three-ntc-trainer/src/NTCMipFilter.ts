/** Separable, repeat-wrapped Lanczos-3 downsampling in physical channel space.
 * No range clamping: signed channels and the filter's negative lobes survive.
 */
export interface NTCSourceMip {
  width: number;
  height: number;
  data: Float32Array;
}
const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));
function weights(source: number, target: number, pixel: number) {
  const scale = source / target,
    center = (pixel + 0.5) * scale - 0.5;
  const taps: [number, number][] = [];
  let sum = 0;
  for (let i = Math.ceil(center - 3 * scale); i <= Math.floor(center + 3 * scale); i++) {
    const d = (i - center) / scale;
    const w = Math.abs(d) < 3 ? sinc(d) * sinc(d / 3) : 0;
    taps.push([((i % source) + source) % source, w]);
    sum += w;
  }
  return taps.map(([i, w]) => [i, w / sum] as [number, number]);
}
export function createLanczosMipChain(
  data: Float32Array,
  width: number,
  height: number,
  channels = 4,
): NTCSourceMip[] {
  if (
    !Number.isInteger(width) ||
    width < 1 ||
    !Number.isInteger(height) ||
    height < 1 ||
    !Number.isInteger(channels) ||
    channels < 1 ||
    data.length !== width * height * channels
  )
    throw new Error("Invalid source mip dimensions or channel count");
  const mips = [{ data: data.slice(), width, height }];
  while (width > 1 || height > 1) {
    const w = Math.max(1, Math.floor(width / 2)),
      h = Math.max(1, Math.floor(height / 2));
    const horizontal = new Float32Array(w * height * channels),
      out = new Float32Array(w * h * channels);
    const wx = Array.from({ length: w }, (_, x) => weights(width, w, x));
    const wy = Array.from({ length: h }, (_, y) => weights(height, h, y));
    for (let y = 0; y < height; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < channels; c++) {
          let sum = 0;
          for (const [i, weight] of wx[x]) sum += data[(y * width + i) * channels + c] * weight;
          horizontal[(y * w + x) * channels + c] = sum;
        }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        for (let c = 0; c < channels; c++) {
          let sum = 0;
          for (const [i, weight] of wy[y]) sum += horizontal[(i * w + x) * channels + c] * weight;
          out[(y * w + x) * channels + c] = sum;
        }
    mips.push({ data: out, width: w, height: h });
    data = out;
    width = w;
    height = h;
  }
  return mips;
}
