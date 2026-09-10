import {
  DataTexture,
  HalfFloatType,
  RGBAFormat,
  NoColorSpace,
  RepeatWrapping,
  LinearFilter,
  LinearMipmapLinearFilter,
} from "three";
import { textureLevel, uv } from "three/tsl";
import { float16ToFloat32, float32ToFloat16 } from "three-ntc";
import { bakeColorNodeToTexture } from "./NTCTextureSource.js";
import { createLanczosMipChain } from "./NTCMipFilter.js";

/** Makes an owned Lanczos mip chain from a square GPU source. Source mip zero
 * is copied in linear/physical units; input textures remain owned by the caller.
 */
export async function createLanczosSourceTexture(
  renderer: any,
  source: any,
): Promise<InstanceType<typeof DataTexture>> {
  const width = source.image?.width,
    height = source.image?.height;
  if (!width || width !== height) throw new Error("Lanczos GPU source requires a square texture");
  const target = await bakeColorNodeToTexture(renderer, textureLevel(source, uv(), 0), width);
  try {
    const raw = await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height);
    const half = raw instanceof Uint16Array ? raw : new Uint16Array(raw.buffer ?? raw);
    // WebGPU readback pads rows to a 256-byte stride. Some backends unpad it.
    const padded = Math.ceil((width * 8) / 256) * 128;
    const stride = half.length === width * height * 4 ? width * 4 : padded;
    const data = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width * 4; x++)
        data[y * width * 4 + x] = float16ToFloat32(half[y * stride + x]);
    const mips = createLanczosMipChain(data, width, height).map((m) => ({
      width: m.width,
      height: m.height,
      data: Uint16Array.from(m.data, float32ToFloat16),
    }));
    const texture = new DataTexture(mips[0].data, width, height, RGBAFormat, HalfFloatType);
    texture.mipmaps = mips;
    texture.generateMipmaps = false;
    texture.colorSpace = NoColorSpace;
    texture.wrapS = texture.wrapT = RepeatWrapping;
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearMipmapLinearFilter;
    texture.needsUpdate = true;
    return texture;
  } finally {
    target.dispose();
  }
}
