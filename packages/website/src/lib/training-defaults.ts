import { getNTCProfileControls } from 'three-ntc-trainer';

/** Shared with the benchmark so measured website defaults cannot drift from the form. */
export const WEBSITE_TRAINING_DEFAULTS = {
  bakeResolution: 1024,
  preset: 'mobile-fast',
  ...getNTCProfileControls('mobile-fast')!,
  positionalEncoding: true,
  dualGrid: true,
  batchSize: 8192,
  iterations: 10000,
  learningRate: 0.01,
  quantization: 'uint2' as const,
};
