// See packages/three-ntc/src/three-shims.d.ts for the rationale: this `three`
// version ships no TypeScript declarations, so every import from 'three' /
// 'three/tsl' / 'three/webgpu' resolves to `any`.
declare module 'three';
declare module 'three/tsl';
declare module 'three/webgpu';

// Custom vitest browser-mode commands registered in the root
// `vitest.config.ts` (`test.projects[].test.browser.commands`) - not part of
// vitest's own `BrowserCommands` interface, so this trainer's `*.gpu.test.ts`
// files (which call `commands.benchmarkConfig()`/`commands.recordMetric()`)
// augment it here rather than casting through `any` at every call site.
declare module 'vitest/browser' {
  interface BrowserCommands {
    /** Server-side benchmark knobs (`NTC_BENCH_*` env vars) - see `vitest.config.ts`. */
    benchmarkConfig: () => Promise<{ iterations: number; physical: boolean; period: number }>;
    /** Logs a `NTC_METRIC <json>` line the CI benchmark harness scrapes from browser console output. */
    recordMetric: (metric: Record<string, unknown>) => Promise<void>;
  }
}
