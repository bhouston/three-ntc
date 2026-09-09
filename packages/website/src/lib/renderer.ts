import { WebGPURenderer } from 'three/webgpu';

// Shared renderer instance (WebGPU compute/render context most of this
// site's material/training code needs). Created lazily, once, in the
// browser - every page that needs a renderer imports this instead of
// constructing its own.
let sharedRenderer: any | null = null;

export function getSharedRenderer() {
  sharedRenderer ??= new WebGPURenderer();
  return sharedRenderer;
}
