import { sin, textureLevel, uv, vec4 } from "three/tsl";
import { expect, it } from "vitest";
import { getRenderer, renderNodeToFloats } from "../../../test/gpu-helpers.js";
import { bakeColorNodeToTexture } from "./NTCTextureSource.js";
import { createLanczosSourceTexture } from "./NTCLanczosSource.js";
import { createLanczosMipChain } from "./NTCMipFilter.js";
import { NTCTrainer } from "./NTCTrainer.js";

it("GPU Lanczos mip uploads match the CPU filter and train without changing source ownership", async () => {
  const renderer = await getRenderer();
  const source = await bakeColorNodeToTexture(
    renderer,
    vec4(sin(uv().x.mul(12.56)), uv().y, -0.5, 0.3),
    16,
  );
  const base = await renderNodeToFloats(renderer, textureLevel(source.texture, uv(), 0), 16);
  const expected = createLanczosMipChain(base, 16, 16);
  const texture = await createLanczosSourceTexture(renderer, source.texture);
  for (let lod = 0; lod < expected.length; lod++) {
    const actual = await renderNodeToFloats(
      renderer,
      textureLevel(texture, uv(), lod),
      expected[lod].width,
    );
    const error = Math.max(...actual.map((x, i) => Math.abs(x - expected[lod].data[i])));
    expect(error).toBeLessThan(0.001);
  }
  const result = await new NTCTrainer({
    mipFilter: "lanczos",
    iterations: 4,
    batchSize: 128,
    baseResolution: 4,
    levels: 1,
    hiddenSizes: [4],
    outputChannels: 4,
  }).train({ renderer, sourceTexture: source.texture, onProgress: () => {} });
  expect(result.iteration).toBe(4);
  expect(Number.isFinite(result.loss)).toBe(true);
  const after = await renderNodeToFloats(renderer, textureLevel(source.texture, uv(), 0), 16);
  expect(after).toEqual(base);
  texture.dispose();
  source.dispose();
});
