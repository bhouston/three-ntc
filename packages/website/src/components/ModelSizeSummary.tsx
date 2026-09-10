import type { NTCChannelClassification } from 'three-ntc';
import { configuredModelInfo } from '../lib/model-info.js';
import type { ModelSizeSettings } from '../lib/model-size.js';
import { ModelInfo } from './ModelInfo.js';

export function ModelSizeSummary({ settings, outputChannels, name = 'Untitled', classification }: {
  settings: ModelSizeSettings; outputChannels: number; name?: string; classification?: NTCChannelClassification | null;
}) {
  return <ModelInfo info={configuredModelInfo(settings, outputChannels, name, classification)} samplingMode={settings.samplingMode} />;
}
