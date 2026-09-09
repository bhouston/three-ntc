// Ported from three.js-v2-basic-ntc's test/unit/addons/ntc/NTC.tests.js
// (QUnit -> vitest), the branch this package's source was originally lifted
// from.
import { describe, expect, it } from 'vitest';

import { activate, createMLP, forwardMLP, powerLog, sigmoid } from './NTCMLP.js';

describe('NTCMLP', () => {
  it('evaluates activations and forward propagation', () => {
    expect(activate(2.5, 'relu')).toBe(2.5);
    expect(activate(-1.5, 'relu')).toBe(0);
    expect(Math.abs(sigmoid(0) - 0.5)).toBeLessThanOrEqual(1e-6);
    expect(powerLog(1, 3)).toBe(0);

    const mlp = createMLP(2, [3], 1, () => 0.75, 'relu', 'linear');
    const run = forwardMLP(mlp, [1, 0.5]);

    expect(run.activations.length).toBe(3); // input, hidden, output
    expect(run.output.length).toBe(1);
  });
});
