// Custom browser commands registered in vitest.config.ts (test.browser.commands).
import 'vitest/browser';

declare module 'vitest/browser' {
  interface BrowserCommands {
    recordMetric(metric: Record<string, unknown>): Promise<void>;
    benchmarkConfig(): Promise<{ iterations: number; physical: boolean; period: number }>;
  }
}
