import { readFileSync, writeFileSync, renameSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import { webdriverio } from '@vitest/browser-webdriverio';

const safari = process.env.NTC_BROWSER === 'safari';
const webkit = process.env.NTC_BROWSER === 'webkit';
const headed = process.env.NTC_GPU_HEADED === '1';
// The chromium channel selects the full browser, not chromium-headless-shell.
const chromiumChannel = process.env.NTC_BROWSER_CHANNEL ?? (process.platform === 'win32' ? 'chromium' : undefined);

const root = fileURLToPath(new URL('.', import.meta.url));

// WebGPU in headless Chromium: the 'new' headless mode keeps the GPU
// process. The default Vulkan launch uses SwiftShader on this Mac. Set
// NTC_GPU_BACKEND=metal to test Apple hardware explicitly; profiles record
// the actual adapter so software/hardware timings cannot be confused.
const chromiumWebGPUArgs = [
  ...(!headed ? ['--headless=new'] : []),
  '--enable-gpu',
  '--enable-unsafe-webgpu',
  ...(process.env.NTC_GPU_BACKEND === 'metal' ? ['--use-angle=metal']
    : process.env.NTC_GPU_BACKEND === 'native' || process.platform === 'win32' ? [] : ['--enable-features=Vulkan']),
  '--ignore-gpu-blocklist',
  ...(process.env.CI ? ['--use-angle=swiftshader', '--use-vulkan=swiftshader'] : []),
];

export default defineConfig({
  optimizeDeps: { include: ['three/addons/loaders/HDRLoader.js', 'three/addons/controls/OrbitControls.js', 'react', 'react-dom/client', 'react/jsx-runtime', '@tanstack/react-form'] },
  resolve: {
    // Tests (both projects) run against package sources, not dist builds.
    alias: {
      '@tanstack/react-form': `${root}packages/website/node_modules/@tanstack/react-form`,
      'react-dom': `${root}packages/website/node_modules/react-dom`,
      'react': `${root}packages/website/node_modules/react`,
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
          fileParallelism: !safari,
          include: ['packages/**/*.gpu.test.ts', ...(process.env.NTC_BENCHMARK_CASE ? ['benchmarks/ntc/run-case.gpu.test.ts'] : [])],
          exclude: ['**/node_modules/**', '**/dist/**'],
          testTimeout: 120_000,
          hookTimeout: 120_000,
          browser: {
            commands: {
              ntcBenchmarkCase: async () => JSON.parse(readFileSync(process.env.NTC_BENCHMARK_CASE!, 'utf8')),
              ntcBenchmarkProgress: async (_context, progress) => {
                const path=process.env.NTC_BENCHMARK_RESULT!+'.progress.json';
                writeFileSync(path,JSON.stringify(progress));
                console.log('NTC_PROGRESS '+JSON.stringify(progress));
              },
              ntcBenchmarkRecord: async (_context, record) => {
                const path=process.env.NTC_BENCHMARK_RESULT!;
                writeFileSync(path+'.tmp',JSON.stringify(record)); renameSync(path+'.tmp',path);
              }, benchmarkConfig: async () => ({iterations:Number(process.env.NTC_BENCH_ITERATIONS || 420),physical:process.env.NTC_BENCH_PHYSICAL === '1',period:Number(process.env.NTC_BENCH_PE_PERIOD || 0)}), recordMetric: async (_context, metric) => { console.log('NTC_METRIC ' + JSON.stringify(metric)); } },
            enabled: true,
            headless: !safari && !headed,
            screenshotFailures: false,
            provider: safari
              ? webdriverio({ logLevel: 'error', connectionRetryCount: 0, connectionRetryTimeout: 60_000 })
              : playwright({
                  launchOptions: { ...(webkit ? {} : { channel: chromiumChannel }), args: webkit ? [] : chromiumWebGPUArgs },
                  contextOptions: { deviceScaleFactor: Number(process.env.NTC_DEVICE_SCALE_FACTOR || 1) },
                }),
            instances: [{ browser: safari ? 'safari' : webkit ? 'webkit' : 'chromium' }],
          },
        },
      },
    ],
  },
});
