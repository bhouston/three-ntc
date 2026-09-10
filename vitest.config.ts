import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

const root = fileURLToPath(new URL('.', import.meta.url));

// WebGPU in headless Chromium: the 'new' headless mode keeps the GPU
// process; SwiftShader (software Vulkan) is what CI runners without a GPU
// fall back to. `--use-angle` is left to Chromium's default so a real GPU
// (Metal on macOS) is used when present.
const chromiumWebGPUArgs = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--ignore-gpu-blocklist',
  ...(process.env.CI ? ['--use-angle=swiftshader', '--use-vulkan=swiftshader'] : []),
];

export default defineConfig({
  resolve: {
    // Tests (both projects) run against package sources, not dist builds.
    alias: {
      'three-ntc-trainer': `${root}packages/three-ntc-trainer/src/index.ts`,
      'three-ntc': `${root}packages/three-ntc/src/index.ts`,
    },
  },
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          include: ['packages/**/*.test.ts'],
          exclude: ['**/node_modules/**', '**/dist/**', '**/*.gpu.test.ts'],
        },
      },
      {
        extends: true,
        test: {
          name: 'gpu',
          include: ['packages/**/*.gpu.test.ts'],
          exclude: ['**/node_modules/**', '**/dist/**'],
          testTimeout: 120_000,
          hookTimeout: 120_000,
          browser: {
            commands: { benchmarkConfig: async () => ({iterations:Number(process.env.NTC_BENCH_ITERATIONS || 420),physical:process.env.NTC_BENCH_PHYSICAL === '1',period:Number(process.env.NTC_BENCH_PE_PERIOD || 0)}), recordMetric: async (_context, metric) => { console.log('NTC_METRIC ' + JSON.stringify(metric)); } },
            enabled: true,
            headless: true,
            screenshotFailures: false,
            provider: playwright({ launchOptions: { args: chromiumWebGPUArgs } }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
