import { formatBytes, formatFlops } from 'three-ntc-trainer';
import { estimateModelSize, type ModelSizeSettings } from '../lib/model-size.js';

export function ModelSizeSummary({ settings, outputChannels }: { settings: ModelSizeSettings; outputChannels: number }) {
  const size = estimateModelSize(settings, outputChannels);
  const bytes = (n: number) => `${formatBytes(n)} (${n.toLocaleString('en-US')} bytes)`;
  return <div className="space-y-2 text-xs text-muted-foreground" aria-label="Model size estimate">
    <dl className="space-y-1">
      <dt>Encoded payload</dt><dd>{bytes(size.storageBytes)}</dd>
      <dt>Runtime estimate</dt><dd>{bytes(size.memoryBytes)}</dd>
      <dt>Decoder</dt><dd>{size.inputSize} inputs · {size.mlpParams.toLocaleString('en-US')} parameters</dd>
      <dt>MLP work per sample</dt><dd>{formatFlops(size.flops)} · {size.decoderEvaluations} {size.decoderEvaluations === 1 ? 'evaluation' : 'evaluations'}</dd>
    </dl>
    <p>FLOPs count dense-layer arithmetic, excluding feature sampling, activations, and shading.</p>
    <p>Estimate for the current settings. Excludes JSON/base64 overhead, GPU padding, and training buffers.</p>
    {settings.quantization === 'none' && <p>Training quantization is off; export still stores 8-bit grids.</p>}
  </div>;
}
