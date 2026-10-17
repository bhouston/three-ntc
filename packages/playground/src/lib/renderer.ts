import { WebGPURenderer } from 'three/webgpu';

// 'three/webgpu' ships no type declarations, so `WebGPURenderer` only
// resolves as a value; `InstanceType<typeof ...>` recovers the instance type.
export type SharedRenderer = InstanceType<typeof WebGPURenderer>;

// Shared renderer instance (WebGPU compute/render context most of this
// site's material/training code needs). Created + initialized lazily, once,
// in the browser - every page that needs a renderer awaits this instead of
// constructing/initializing its own.
let sharedRendererInit: Promise<SharedRenderer> | null = null;

export function getSharedRenderer(): Promise<SharedRenderer> {
  sharedRendererInit ??= (async () => {
    const renderer = new WebGPURenderer({ antialias: false });
    await renderer.init();
    return renderer;
  })();
  return sharedRendererInit;
}
