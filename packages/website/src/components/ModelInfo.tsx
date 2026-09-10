import type { NTCSamplingMode } from 'three-ntc';
import { formatBytes, formatFlops } from 'three-ntc-trainer';
import type { ModelInfoData } from '../lib/model-info.js';

export function ModelInfo({ info, samplingMode }: { info: ModelInfoData; samplingMode: NTCSamplingMode }) {
  const {size} = info;
  const evaluations = samplingMode === 'trilinear' ? 8 : 1;
  const bytes = (n: number) => `${formatBytes(n)} (${n.toLocaleString('en-US')} bytes)`;
  return <div className="space-y-3 text-xs text-muted-foreground" aria-label="Model Info">
    <dl className="space-y-1">
      <dt>Name</dt><dd>{info.name}</dd>
      <dt>Encoded payload</dt><dd>{bytes(size.storageBytes)}</dd>
      <dt>Runtime estimate</dt><dd>{bytes(size.memoryBytes)}</dd>
      <dt>Latent grids</dt><dd className="break-words">{info.grids.length} ({info.grids.map(g => `${g.group}: ${g.width}×${g.height}×${g.channels} @ ${g.bits}-bit`).join(', ')})</dd>
      <dt>MLP layers</dt><dd>{info.layers.length} ({[info.layers[0]?.inputSize, ...info.layers.map(layer => layer.outputSize)].join('→')})</dd>
      <dt>Decoder</dt><dd>{size.inputSize} inputs · {size.mlpParams.toLocaleString('en-US')} parameters</dd>
      <dt>MLP work per sample</dt><dd>{formatFlops(size.flopsPerDecode * evaluations)} · {evaluations} {evaluations === 1 ? 'evaluation' : 'evaluations'}</dd>
    </dl>
    {info.trainingQuantizationOff && <p>Training quantization is off; export still stores 8-bit grids.</p>}
    <details>
      <summary className="cursor-pointer font-medium text-foreground">Channels ({info.activeChannels.length})</summary>
      <table className="mt-2 w-full text-left text-xs">
        <thead><tr><th className="pr-4 font-normal">Channel</th><th className="font-normal">Source</th></tr></thead>
        <tbody>
          {info.activeChannels.map(key => <tr key={key}><td className="pr-4">{key}</td><td>MLP</td></tr>)}
          {info.constantChannels.map(key => <tr key={key}><td className="pr-4">{key}</td><td>Fixed</td></tr>)}
        </tbody>
      </table>
    </details>
  </div>;
}
