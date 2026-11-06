import { expect, it, vi } from 'vitest';
import { resolveGradientPrecision, validateGPUDispatch } from './NTCGPUCapabilities.js';

function device(compiles = true) {
  const mocks = {
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn().mockResolvedValue(null),
    createShaderModule: vi.fn().mockReturnValue({}),
    createComputePipelineAsync: compiles
      ? vi.fn().mockResolvedValue({})
      : vi.fn().mockRejectedValue(new Error('field may not be qualified with an address space')),
  };
  // The mocks stay reachable un-bound on `.mocks`; the object itself stands in for a GPUDevice.
  return Object.assign(mocks as unknown as GPUDevice, { mocks });
}

it('falls back before training when the Safari atomic shader cannot compile', async () => {
  const gpu = device(false);
  const warning = vi.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    expect(await resolveGradientPrecision(gpu, 'auto')).toBe('fixed');
    expect(gpu.mocks.popErrorScope).toHaveBeenCalledOnce();
    await expect(resolveGradientPrecision(gpu, 'float')).rejects.toThrow('cannot compile');
    expect(gpu.mocks.createComputePipelineAsync).toHaveBeenCalledOnce();
  } finally {
    warning.mockRestore();
  }
});

it('uses float when compilation succeeds and caches support per device', async () => {
  const gpu = device();
  expect(await resolveGradientPrecision(gpu, 'auto')).toBe('float');
  expect(await resolveGradientPrecision(gpu, 'float')).toBe('float');
  expect(gpu.mocks.createComputePipelineAsync).toHaveBeenCalledOnce();
});

it('does not compile the unsupported helper when fixed is requested', async () => {
  const gpu = device(false);
  expect(await resolveGradientPrecision(gpu, 'fixed')).toBe('fixed');
  expect(gpu.mocks.createShaderModule).not.toHaveBeenCalled();
});

it('turns invalid GPU dispatches into a training error instead of zero loss', async () => {
  const gpu = device();
  gpu.mocks.popErrorScope.mockResolvedValue({ message: 'GPUComputePipeline is invalid' });
  await expect(validateGPUDispatch(gpu, () => {})).rejects.toThrow('training GPU dispatch failed');
});

it('balances validation scopes when dispatch throws synchronously', async () => {
  const gpu = device();
  await expect(
    validateGPUDispatch(gpu, () => {
      throw new Error('dispatch failed');
    }),
  ).rejects.toThrow('dispatch failed');
  expect(gpu.mocks.popErrorScope).toHaveBeenCalledOnce();
});
