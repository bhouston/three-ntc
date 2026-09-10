import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it } from 'vitest';
import { ModelInfo } from './ModelInfo.js';
import type { ModelInfoData } from '../lib/model-info.js';

it('starts Channels collapsed, counts only active channels, and expands the shared channel table', async () => {
  const info: ModelInfoData = {
    name:'Example', size:{storageBytes:100, memoryBytes:200, mlpParams:20, inputSize:5, flopsPerDecode:40},
    grids:[{group:'G0',width:4,height:4,channels:4,bits:4}], layers:[{inputSize:5,outputSize:5}],
    activeChannels:['albedo','normal'], constantChannels:['metalness'],
  };
  const fixture=document.createElement('div');document.body.append(fixture);
  const root=createRoot(fixture);
  try {
    root.render(createElement(ModelInfo,{info,samplingMode:'nearest'}));
    await expect.poll(()=>fixture.querySelector('summary')?.textContent).toBe('Channels (2)');
    const details=fixture.querySelector('details')!;
    expect(details.open).toBe(false);
    expect(fixture.textContent).not.toContain('Estimate for the current settings');
    fixture.querySelector('summary')!.click();
    expect(details.open).toBe(true);
    expect([...fixture.querySelectorAll('tbody tr')].map(row=>[...row.querySelectorAll('td')].map(cell=>cell.textContent)))
      .toEqual([['albedo','MLP'],['normal','MLP'],['metalness','Fixed']]);
    root.render(createElement(ModelInfo,{info,samplingMode:'trilinear'}));
    await expect.poll(()=>fixture.textContent).toContain('320 FLOP · 8 evaluations');
    expect(details.open).toBe(true);
    fixture.querySelector('summary')!.click();
    expect(details.open).toBe(false);
  } finally {root.unmount();fixture.remove();}
});
