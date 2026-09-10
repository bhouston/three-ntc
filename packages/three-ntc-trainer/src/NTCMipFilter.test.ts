import { expect, it } from "vitest";
import { createLanczosMipChain } from "./NTCMipFilter.js";

it("preserves signed DC channels through odd and rectangular mip chains", () => {
  const data = Float32Array.from({ length: 7 * 5 * 2 }, (_, i) => (i % 2 ? 2 : -0.75));
  const mips = createLanczosMipChain(data, 7, 5, 2);
  expect(mips.map((m) => [m.width, m.height])).toEqual([
    [7, 5],
    [3, 2],
    [1, 1],
  ]);
  for (const mip of mips)
    for (let i = 0; i < mip.data.length; i++) expect(mip.data[i]).toBeCloseTo(i % 2 ? 2 : -0.75, 6);
  expect(mips[0].data).not.toBe(data);
});

for (const frequency of [1 / 8, 3 / 8]) {
  it(`reduces ${frequency < 0.25 ? "passband attenuation" : "aliasing"} versus box mipmaps`, () => {
    const size = 64;
    const signal = (x: number) => Math.cos(2 * Math.PI * frequency * x);
    const data = Float32Array.from({ length: size * size }, (_, i) => signal(i % size));
    const mip = createLanczosMipChain(data, size, size, 1)[1];
    let boxMse = 0,
      lanczosMse = 0;
    for (let y = 0; y < 32; y++)
      for (let x = 0; x < 32; x++) {
        // Ideal low-pass reconstruction keeps the passband sinusoid and removes
        // frequencies above the new Nyquist limit; no arbitrary PSNR threshold.
        const ideal = frequency < 0.25 ? signal(2 * x + 0.5) : 0;
        boxMse += ((signal(2 * x) + signal(2 * x + 1)) / 2 - ideal) ** 2 / 1024;
        lanczosMse += (mip.data[y * 32 + x] - ideal) ** 2 / 1024;
      }
    expect(lanczosMse).toBeLessThan(boxMse);
    console.log("NTC_MIP_METRIC", JSON.stringify({ frequency, boxMse, lanczosMse }));
  });
}
