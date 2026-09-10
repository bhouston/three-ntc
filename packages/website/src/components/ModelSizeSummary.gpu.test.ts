// Uses the existing browser project; no GPU is needed for this form regression.
import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { useForm, useStore } from '@tanstack/react-form';
import { expect, it } from 'vitest';
import { SamplingModeSelect } from './SamplingModeSelect.js';
import { ModelSizeSummary } from './ModelSizeSummary.js';
import type { ModelSizeSettings } from '../lib/model-size.js';

it('updates the visible card as the trainer form changes and restores values when toggled back', async () => {
  function Harness() {
    const form = useForm({defaultValues: {
      baseResolution: 16, levels: 2, hiddenSize: 16, hiddenLayers: 1,
      positionalEncoding: true, dualGrid: true, quantization: 'uint8', samplingMode: 'nearest',
    } as ModelSizeSettings});
    const settings = useStore(form.store, state => state.values);
    return createElement('div', null,
      ...(['positionalEncoding', 'dualGrid'] as const).map(key => createElement('button', {
        key, onClick: () => form.setFieldValue(key, !form.getFieldValue(key)),
      }, key)),
      ...(['uint8', 'uint4', 'uint2', 'none'] as const).map(mode => createElement('button', {
        key: mode, onClick: () => form.setFieldValue('quantization', mode),
      }, mode)),
      createElement(SamplingModeSelect, {value: settings.samplingMode, onChange: mode => form.setFieldValue('samplingMode', mode)}),
      createElement(ModelSizeSummary, { settings, outputChannels: 3 }),
    );
  }
  const fixture = document.createElement('div');
  document.body.append(fixture);
  const root = createRoot(fixture);
  const click = (label: string) => [...fixture.querySelectorAll('button')].find(button => button.textContent === label)!.click();
  const payload = () => [...fixture.querySelectorAll('dt')].find(term => term.textContent === 'Encoded payload')?.nextElementSibling?.textContent;
  try {
    root.render(createElement(Harness));
    await expect.poll(payload).toContain('2,550 bytes');
    const work = () => [...fixture.querySelectorAll('dt')].find(term => term.textContent === 'MLP work per sample')?.nextElementSibling?.textContent;
    await expect.poll(work).toBe('1.2K FLOP · 1 evaluation');
    const select = fixture.querySelector('select')!;
    for (const [mode, expected] of [['trilinear', '9.4K FLOP · 8 evaluations'], ['stochastic', '1.2K FLOP · 1 evaluation'], ['nearest', '1.2K FLOP · 1 evaluation']]) {
      select.value = mode;
      select.dispatchEvent(new Event('change', {bubbles: true}));
      await expect.poll(work).toBe(expected);
      expect(payload()).toContain('2,550 bytes');
    }
    click('positionalEncoding');
    await expect.poll(payload).toContain('1,782 bytes');
    click('positionalEncoding');
    await expect.poll(payload).toContain('2,550 bytes');
    click('dualGrid');
    await expect.poll(payload).toContain('2,150 bytes');
    click('dualGrid');
    await expect.poll(payload).toContain('2,550 bytes');
    for (const [mode, bytes] of [['uint4', '1,870'], ['uint2', '1,530'], ['uint8', '2,550'], ['none', '2,550']]) {
      click(mode);
      await expect.poll(payload).toContain(`${bytes} bytes`);
    }
    await expect.poll(() => fixture.textContent).toContain('export still stores 8-bit grids');
  } finally {
    root.unmount();
    fixture.remove();
  }
});
