/** Nearest-rank percentiles; keep raw observations alongside these summaries. */
export function summarizeTimings(values: number[]) {
  if (!values.length || values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("Timing samples must be nonempty, finite, and nonnegative");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (p: number) => sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)]!;
  return {
    count: values.length,
    meanMs: values.reduce((sum, value) => sum + value, 0) / values.length,
    p50Ms: percentile(0.5),
    p95Ms: percentile(0.95),
    p99Ms: percentile(0.99),
    maxMs: sorted[sorted.length - 1]!,
  };
}
