import { expect, it } from "vitest";
import { summarizeTimings } from "../../../../test/performance-metrics.js";

it("preserves tail stalls in timing summaries without mutating samples", () => {
  const values = [1000, ...Array(99).fill(10)];
  expect(summarizeTimings(values)).toEqual({
    count: 100,
    meanMs: 19.9,
    p50Ms: 10,
    p95Ms: 10,
    p99Ms: 10,
    maxMs: 1000,
  });
  expect(values[0]).toBe(1000);
  expect(summarizeTimings([20, 0, 10, 30])).toEqual({
    count: 4, meanMs: 15, p50Ms: 10, p95Ms: 30, p99Ms: 30, maxMs: 30,
  });
});

it("rejects missing or invalid observations instead of reporting a successful profile", () => {
  for (const values of [[], [NaN], [Infinity], [-1]]) {
    expect(() => summarizeTimings(values)).toThrow();
  }
});
