// Custom browser commands registered in vitest.config.ts (test.browser.commands).
//
// `vitest/browser` re-exports `@vitest/browser-<provider>/context` for every
// provider package. With both playwright and webdriverio installed, each
// resolves a separate copy of `@vitest/browser`, so `export *` sees `commands`
// twice and TypeScript drops it as ambiguous. Declaring it here makes it explicit.
import 'vitest/browser';

declare module 'vitest/browser' {
  interface BrowserCommands {
    recordMetric(metric: Record<string, unknown>): Promise<void>;
    benchmarkConfig(): Promise<{ iterations: number; physical: boolean; period: number }>;
  }

  export const commands: BrowserCommands;
}
