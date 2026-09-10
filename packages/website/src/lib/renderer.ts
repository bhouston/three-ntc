import { WebGPURenderer } from 'three/webgpu';

// Shared renderer instance (WebGPU compute/render context most of this
// site's material/training code needs). Created + initialized lazily, once,
// in the browser - every page that needs a renderer awaits this instead of
// constructing/initializing its own.
let sharedRendererInit: Promise<any> | null = null;

export function getSharedRenderer(): Promise<any> {
  sharedRendererInit ??= (async () => {
    const renderer = new WebGPURenderer({ antialias: false });
    await renderer.init();
    return renderer;
  })();
  return sharedRendererInit;
}
