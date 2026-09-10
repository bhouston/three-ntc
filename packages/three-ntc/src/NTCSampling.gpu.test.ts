import { HalfFloatType, Mesh, OrthographicCamera, PlaneGeometry, RenderTarget, Scene } from "three";
import { float, floor, int, uv, vec2, vec3, vec4 } from "three/tsl";
import { expect, it } from "vitest";
import { commands } from "vitest/browser";
import { evaluateNeuralTextureSampled } from "./NTCDecoderTSL.js";
import { NTCNodeMaterial } from "./NTCNodeMaterial.js";
import { buildLevelTextures } from "./NTCHalfFloatTexture.js";
import { CHANNELS, getChannel, layoutChannels } from "./NTCFormat.js";
import {
  evaluateNTCCpu,
  getRenderer,
  makeModel,
  readRenderTargetFloats,
  renderNodeToFloats,
} from "../../../test/gpu-helpers.js";

function fixture() {
  const model = makeModel(1, {
    gridChannels: 1,
    levels: 1,
    baseResolution: 4,
    textureResolution: 4,
    hiddenSizes: [],
    outputChannels: 3,
  });
  model.grids[0].data.set(Array.from({ length: 16 }, (_, i) => (i - 8) / 4));
  const layer = model.decoder.layers[0];
  layer.weights.fill(0);
  layer.biases.fill(0);
  for (let c = 0; c < 3; c++) {
    layer.weights[c * layer.inputSize] = 1;
    layer.weights[c * layer.inputSize + layer.inputSize - 1] = 2;
  }
  return model;
}
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

it("nearest reconstructs exactly one texel at the nearest clamped physical mip", async () => {
  const renderer = await getRenderer(),
    model = fixture(),
    textures = buildLevelTextures(model);
  try {
    for (const [u, v, lod] of [
      [0.4, 0.7, 0.2],
      [-0.1, 1.1, 0.7],
      [0.5, 0.5, -1],
      [0.9, 0.1, 10],
    ]) {
      const level = Math.floor(Math.max(0, Math.min(model.maxLod, lod)) + 0.5);
      const size = Math.max(1, Math.floor(4 / 2 ** level));
      const expected = evaluateNTCCpu(
        model,
        (Math.floor(u * size) + 0.5) / size,
        (Math.floor(v * size) + 0.5) / size,
        level,
      ).map(sigmoid);
      const nodes = evaluateNeuralTextureSampled(vec2(u, v), model, textures, float(lod), [
        "sigmoid",
        "sigmoid",
        "sigmoid",
      ]);
      const actual = await renderNodeToFloats(renderer, vec4(...nodes, 1), 1);
      expected.forEach((value, i) => expect(Math.abs(actual[i] - value)).toBeLessThan(0.001));
    }
  } finally {
    textures.forEach((texture) => texture.dispose());
  }
});

it("stochastic samples individual decoded texels and its stratified mean matches trilinear filtering", async () => {
  const renderer = await getRenderer(),
    model = fixture(),
    textures = buildLevelTextures(model);
  // Exhaust all 4^3 independent, stratified UV/LOD samples across an 8x8 target.
  const index = floor(uv().x.mul(8)).add(floor(uv().y.mul(8)).mul(8));
  const random = vec3(index.mod(4), floor(index.div(4)).mod(4), floor(index.div(16)))
    .add(0.5)
    .div(4);
  const u = 0.5,
    v = 0.5,
    lod = 0.25;
  let expected = 0;
  const possible: number[] = [];
  for (let level = 0; level < 2; level++) {
    const size = 4 / 2 ** level,
      px = u * size - 0.5,
      py = v * size - 0.5;
    for (let y = 0; y < 2; y++)
      for (let x = 0; x < 2; x++) {
        const value = sigmoid(
          evaluateNTCCpu(
            model,
            (Math.floor(px) + x + 0.5) / size,
            (Math.floor(py) + y + 0.5) / size,
            level,
          )[0],
        );
        possible.push(value);
        expected += value * 0.25 * (level ? lod : 1 - lod);
      }
  }
  try {
    const nodes = evaluateNeuralTextureSampled(
      vec2(u, v),
      model,
      textures,
      float(lod),
      ["sigmoid", "sigmoid", "sigmoid"],
      int(1),
      random,
    );
    const actual = await renderNodeToFloats(renderer, vec4(...nodes, 1), 8);
    const values = Array.from({ length: 64 }, (_, i) => actual[i * 4]);
    for (const value of values)
      expect(Math.min(...possible.map((p) => Math.abs(p - value)))).toBeLessThan(0.001);
    expect(new Set(values).size).toBeGreaterThan(1);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(Math.abs(mean - expected)).toBeLessThan(0.001);
    await (commands as any).recordMetric({
      kind: "stochastic-filter-oracle",
      mean,
      expected,
      absoluteError: Math.abs(mean - expected),
    });
  } finally {
    textures.forEach((texture) => texture.dispose());
  }
});

it("switches live material modes without recompilation and bounds nearest/stochastic to one decode", async () => {
  const renderer = await getRenderer(),
    model = fixture();
  const material = new NTCNodeMaterial(
    model,
    {
      activeChannels: layoutChannels([getChannel("albedo")]).channels,
      constantValues: Object.fromEntries(CHANNELS.map((c) => [c.key, c.defaultValue])),
    },
    { debugView: "albedo", lodNode: float(0.25) },
  );
  const geometry = new PlaneGeometry(2, 2),
    scene = new Scene();
  scene.add(new Mesh(geometry, material));
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 4);
  camera.position.z = 2;
  const target = new RenderTarget(32, 32, { type: HalfFloatType }),
    previous = renderer.getRenderTarget();
  const device = renderer.backend.device,
    create = device.createShaderModule.bind(device);
  const shaders: string[] = [];
  device.createShaderModule = (descriptor: any) => {
    shaders.push(descriptor.code);
    return create(descriptor);
  };
  try {
    expect(material.samplingMode).toBe("nearest");
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const nearest = await readRenderTargetFloats(renderer, target, 32),
      initialModules = shaders.length;
    const fragment = shaders.find((code) => code.includes("@fragment"))!;
    // This checks the generated execution bound, not just zero-valued tap weights.
    const branch = fragment.match(/if \( (\w+) \) \{\s*(\w+) = 8;\s*\} else \{\s*\2 = 1;\s*\}/);
    expect(branch).not.toBeNull();
    expect(fragment).toContain(`ntcSampleCount = ${branch![2]};`);
    expect(fragment).toMatch(/for \( var ntcSample[^;]*; ntcSample < ntcSampleCount;/);
    const version = material.version;
    material.setSamplingMode("trilinear");
    renderer.render(scene, camera);
    const trilinear = await readRenderTargetFloats(renderer, target, 32);
    expect(trilinear.some((value, i) => Math.abs(value - nearest[i]) > 0.001)).toBe(true);
    material.samplingMode = "stochastic";
    renderer.render(scene, camera);
    const stochastic = await readRenderTargetFloats(renderer, target, 32);
    // Three advances frameId on animation frames, not on every render pass.
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    renderer.render(scene, camera);
    const nextFrame = await readRenderTargetFloats(renderer, target, 32);
    expect(stochastic.some((value, i) => Math.abs(value - nextFrame[i]) > 0.001)).toBe(true);
    material.setSamplingMode("nearest");
    renderer.render(scene, camera);
    const restored = await readRenderTargetFloats(renderer, target, 32);
    expect(restored).toEqual(nearest);
    expect(shaders).toHaveLength(initialModules);
    expect(material.version).toBe(version);
    expect(() => material.setSamplingMode("invalid" as any)).toThrow();
  } finally {
    device.createShaderModule = create;
    renderer.setRenderTarget(previous);
    target.dispose();
    geometry.dispose();
    material.dispose();
  }
});
