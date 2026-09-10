import { FLOAT_ATOMIC_WGSL } from './NTCFloatAtomic.js';

const floatSupport = new WeakMap<object, Promise<boolean>>();

// WebGPU advertises integer atomics, but some Safari Metal translators fail
// to compile atomicCompareExchangeWeak. Probe the actual accumulation helper,
// rather than guessing support from the user agent or the WebGPU feature list.
function supportsFloatAccumulation(device: any): Promise<boolean> {
  let pending = floatSupport.get(device);
  if (!pending) {
    pending = (async () => {
      device.pushErrorScope('validation');
      let supported = false;
      try {
        const module = device.createShaderModule({ code: `${FLOAT_ATOMIC_WGSL}
@group(0) @binding(0) var<storage, read_write> accumulator: atomic<i32>;
@compute @workgroup_size(1) fn main() {
  let previous = ntcAtomicAddFloat(&accumulator, 1.0);
}` });
        await device.createComputePipelineAsync({
          label: 'NTC float accumulation capability check', layout: 'auto',
          compute: { module, entryPoint: 'main' }
        });
        supported = true;
      } catch {
        // A failed pipeline is expected on affected Safari versions.
      } finally {
        if (await device.popErrorScope()) supported = false;
      }
      return supported;
    })();
    floatSupport.set(device, pending);
  }
  return pending;
}

export async function resolveGradientPrecision(
  device: any, requested: 'auto' | 'fixed' | 'float'
): Promise<'fixed' | 'float'> {
  if (requested === 'fixed') return 'fixed';
  if (requested !== 'auto' && requested !== 'float') throw new Error('Invalid gradientPrecision');
  if (await supportsFloatAccumulation(device)) return 'float';
  if (requested === 'float') {
    throw new Error('NTCTrainer: floating-point gradient accumulation cannot compile on this device. Use gradientPrecision: "auto" or "fixed".');
  }
  console.warn('NTCTrainer: floating-point atomics unavailable; using fixed-point gradient accumulation.');
  return 'fixed';
}

export async function validateGPUDispatch(device: any, dispatch: () => void): Promise<void> {
  device.pushErrorScope('validation');
  try {
    dispatch();
  } finally {
    const error = await device.popErrorScope();
    if (error) throw new Error(`NTCTrainer: training GPU dispatch failed: ${error.message}`);
  }
}
