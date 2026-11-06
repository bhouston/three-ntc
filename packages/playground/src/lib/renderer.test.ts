import { describe, expect, it, vi } from 'vitest';

// Regression test for the "Renderer: .render() called before the backend is
// initialized" bug: getSharedRenderer() used to hand back a freshly
// constructed WebGPURenderer without ever awaiting `.init()`, so the first
// caller to use it (bakeMaterialToTextures in the trainer) would render
// before the backend was ready. getSharedRenderer() must await init() itself
// so every caller gets an already-initialized renderer.
vi.mock('three/webgpu', () => {
  class WebGPURenderer {
    initCalled = false;
    async init() {
      // Simulate async backend setup - if a caller used the renderer
      // before this resolves, that's the bug this test guards against.
      await Promise.resolve();
      this.initCalled = true;
    }
  }
  return { WebGPURenderer };
});

describe('getSharedRenderer', () => {
  it('resolves only once the renderer is initialized', async () => {
    const { getSharedRenderer } = await import('./renderer.js');
    const renderer = await getSharedRenderer();
    expect(renderer.initCalled).toBe(true);
  });

  it('memoizes a single renderer/init across callers', async () => {
    const { getSharedRenderer } = await import('./renderer.js');
    const [a, b] = await Promise.all([getSharedRenderer(), getSharedRenderer()]);
    expect(a).toBe(b);
  });
});
