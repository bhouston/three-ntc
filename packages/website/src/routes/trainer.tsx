import { createFileRoute } from '@tanstack/react-router';
import { useForm, useStore } from '@tanstack/react-form';
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGoogleAnalytics } from 'tanstack-router-ga4';
import * as THREE from 'three';
import { defineChart, lineY } from '@tanstack/charts';
import { scaleLinear } from '@tanstack/charts/scales/linear';
import { Chart } from '@tanstack/charts/react';
import { ChevronDown, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import {
  bakeMaterialToTextures,
  classifyMaterialChannels,
  computeGridLatentTexels,
  computeGridLevels,
  computeMLPFlops,
  computeMLPParamCount,
  computeModelFootprint,
  formatModelSizeSummary,
  GRID_BASE_RESOLUTION_OPTIONS,
  GRID_LEVELS_OPTIONS,
  getNTCProfile,
  inferAlbedoUvTransform,
  MATERIALX_SAMPLES,
  MaterialXLoader,
  MLP_ACTIVATION_OPTIONS,
  MLP_HIDDEN_SIZE_OPTIONS,
  NTC_PROFILE_NAMES,
  NTC_PROFILES,
  NTCExporter,
  NTCTrainer,
} from 'three-ntc-trainer';
import { buildChannelActivations, MAX_TOTAL_CHANNELS, NTCNodeMaterial } from 'three-ntc';

import { NTCViewer, type NTCViewerShape } from '@/components/NTCViewer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Field, FieldError, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { getSharedRenderer } from '@/lib/renderer';
import { cn } from '@/lib/utils';

export const Route = createFileRoute('/trainer')({
  component: TrainerPage,
});

const BAKE_RESOLUTION_OPTIONS = [128, 256, 512, 1024, 2048, 4096];
const BATCH_SIZE_OPTIONS = [1024, 2048, 4096, 8192, 16384];
const QUANTIZATION_OPTIONS: Array<{ value: 'none' | 'uint8'; label: string }> = [
  { value: 'none', label: 'Off' },
  { value: 'uint8', label: '8-bit' },
];
const SHAPE_OPTIONS: NTCViewerShape[] = ['torus', 'sphere', 'plane'];
const DEFAULT_MATERIALX_KEY = 'brick';

type FormValues = {
  bakeResolution: number;
  preset: string;
  levels: number;
  baseResolution: number;
  hiddenSize: number;
  hiddenActivation: string;
  batchSize: number;
  iterations: number;
  learningRate: number;
  quantization: 'none' | 'uint8';
  shape: NTCViewerShape;
  interpolation: boolean;
  lodBias: number;
};

const DEFAULT_VALUES: FormValues = {
  bakeResolution: 1024,
  preset: 'mobile-balanced',
  levels: 3,
  baseResolution: 256,
  hiddenSize: 8,
  hiddenActivation: 'relu',
  batchSize: 8192,
  iterations: 10000,
  learningRate: 0.01,
  quantization: 'uint8',
  shape: 'torus',
  interpolation: true,
  lodBias: 0,
};

function isPhysicalNodeMaterial(material: any): boolean {
  return material !== undefined && (material?.isMeshPhysicalNodeMaterial === true || material?.type === 'MeshPhysicalNodeMaterial');
}

function download(filename: string, text: string) {
  const blob = new Blob([text], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// Small shared render for a <Select> bound to a `form.Field` render-prop
// object - every network/training/object/view dropdown on this page uses
// this same label + trigger + content shape, only the option list/parser
// change.
function SelectFormField({
  field,
  label,
  options,
  getLabel = String,
  disabled,
  parse = (v: string) => v as unknown,
}: {
  field: any;
  label: string;
  options: readonly (string | number)[];
  getLabel?: (value: string | number) => string;
  disabled?: boolean;
  parse?: (value: string) => unknown;
}) {
  return (
    <Field data-invalid={field.state.meta.errors.length > 0}>
      <FieldLabel htmlFor={field.name}>{label}</FieldLabel>
      <Select value={String(field.state.value)} onValueChange={(v) => field.handleChange(parse(v))} disabled={disabled}>
        <SelectTrigger id={field.name} size="sm">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((opt) => (
            <SelectItem key={String(opt)} value={String(opt)}>
              {getLabel(opt)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldError errors={field.state.meta.errors} />
    </Field>
  );
}

// A FieldSet that can collapse its FieldGroup away, legend doubling as the
// toggle button. Defaults open so existing behavior/layout doesn't change
// until someone actually clicks a section shut.
function CollapsibleSection({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Collapsible defaultOpen={defaultOpen} className="rounded-lg border border-border p-4">
      <CollapsibleTrigger className="group -ml-1 flex w-full items-center gap-1 px-1 text-sm font-medium text-foreground">
        <ChevronDown className="size-4 shrink-0 transition-transform group-data-[state=closed]:-rotate-90" />
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-4">
        <FieldGroup>{children}</FieldGroup>
      </CollapsibleContent>
    </Collapsible>
  );
}

function TrainerPage() {
  const ga = useGoogleAnalytics();
  const form = useForm({ defaultValues: DEFAULT_VALUES });
  const values = useStore(form.store, (state) => state.values);

  const [status, setStatus] = useState('Ready.');
  const [settingsOpen, setSettingsOpen] = useState(true);
  const [mtlxDragOver, setMtlxDragOver] = useState(false);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [builtInKey, setBuiltInKey] = useState(DEFAULT_MATERIALX_KEY);
  const [hasSource, setHasSource] = useState(false);
  const [channelClassification, setChannelClassification] = useState<any | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [hasTrainedModel, setHasTrainedModel] = useState(false);
  const [previewMaterial, setPreviewMaterial] = useState<any>(null);
  const [teacherMaterial, setTeacherMaterial] = useState<any>(null);

  const materialXMaterialRef = useRef<any>(null);
  const uvTransformRef = useRef<any>(new THREE.Matrix3());
  const activeTrainerRef = useRef<any>(null);
  const previewMaterialRef = useRef<any>(null);
  const sourceTexturesRef = useRef<any[] | null>(null);
  const mtlxInputRef = useRef<HTMLInputElement>(null);

  // Full-resolution point history lives in a ref (training can call
  // onProgress a couple thousand times over a run) - `lossPoints` state is a
  // throttled snapshot of it, flushed at most every `LOSS_FLUSH_INTERVAL_MS`,
  // so re-rendering the TanStack Charts SVG scene doesn't fight the training
  // loop for every single progress tick.
  const LOSS_FLUSH_INTERVAL_MS = 150;
  const lossPointsRef = useRef<Array<{ iteration: number; loss: number }>>([]);
  const lastLossFlushRef = useRef(0);
  const trainStartRef = useRef(0);
  const [lossPoints, setLossPoints] = useState<Array<{ iteration: number; loss: number; logLoss: number }>>([]);
  const [lossIps, setLossIps] = useState('');

  const resetLoss = useCallback(() => {
    lossPointsRef.current = [];
    lastLossFlushRef.current = 0;
    trainStartRef.current = performance.now();
    setLossPoints([]);
    setLossIps('');
  }, []);

  const addLossPoint = useCallback((point: { iteration: number; loss: number }, force = false) => {
    lossPointsRef.current.push(point);
    const now = performance.now();
    if (!force && now - lastLossFlushRef.current < LOSS_FLUSH_INTERVAL_MS) return;
    lastLossFlushRef.current = now;
    // `loss` is always > 0 (L2 loss) in practice, but guard the log anyway.
    setLossPoints(lossPointsRef.current.map((p) => ({ ...p, logLoss: Math.log10(Math.max(p.loss, 1e-12)) })));
    const elapsedSeconds = (now - trainStartRef.current) / 1000;
    if (elapsedSeconds > 0) setLossIps(`${(point.iteration / elapsedSeconds).toFixed(1)} it/s`);
  }, []);

  const lossChartDefinition = useMemo(
    () =>
      defineChart({
        marks: [lineY(lossPoints, { x: 'iteration', y: 'logLoss', stroke: '#50c8ff', strokeWidth: 2 })],
        scales: {
          x: { scale: scaleLinear, axis: { label: 'Iteration' } },
          y: { scale: scaleLinear, nice: true, grid: true, axis: { label: 'log10(L2 loss)' } },
        },
      }),
    [lossPoints],
  );

  const disposeSourceTextures = useCallback(() => {
    for (const rt of sourceTexturesRef.current ?? []) rt.dispose();
    sourceTexturesRef.current = null;
  }, []);

  const rebuildPreviewMaterial = useCallback((renderer: any, cpuModel: any, classification: any, lodBias: number) => {
    const previous = previewMaterialRef.current;
    const material = new NTCNodeMaterial(cpuModel, classification, {
      renderer,
      // ponytail: lodBias is only baked in at construction time (a plain
      // number, not a live TSL uniform) - the simplest option
      // NTCNodeMaterial's constructor supports. Dragging the slider takes
      // effect on the next train/rebuild rather than instantly; upgrade path
      // is passing a shared `uniform(lodBias)` node here and writing its
      // `.value` on slider change instead.
      lodBias,
      interpolation: form.getFieldValue('interpolation'),
    });
    previewMaterialRef.current = material;
    setPreviewMaterial(material);
    setHasTrainedModel(cpuModel !== null);
    if (previous?.dispose) previous.dispose();
    return material;
  }, [form]);

  const setSourceMaterial = useCallback((material: any, name: string) => {
    materialXMaterialRef.current = material;
    setTeacherMaterial(material);
    const classification = classifyMaterialChannels(material);
    setChannelClassification(classification);

    const surfaceShaderNode = material.materialXSurfaceShaderNode;
    const materialXDocument = material.materialXDocument;
    uvTransformRef.current =
      surfaceShaderNode && materialXDocument
        ? inferAlbedoUvTransform(materialXDocument, surfaceShaderNode)
        : new THREE.Matrix3();

    setSourceName(material.name || name);
    setHasSource(true);

    if (previewMaterialRef.current?.dispose) previewMaterialRef.current.dispose();
    previewMaterialRef.current = null;
    setPreviewMaterial(null);
    setHasTrainedModel(false);
    disposeSourceTextures();

    const activeKeys = classification.activeChannels.map((c: any) => c.key).join(', ') || 'none';
    const uvNote = uvTransformRef.current.equals(new THREE.Matrix3())
      ? ''
      : ' A UV transform was detected on the albedo graph and will be baked out / re-applied at render time.';
    setStatus(
      `MaterialX loaded. Training ${classification.totalChannels}/${MAX_TOTAL_CHANNELS} channels: ${activeKeys}.${uvNote} Press Train to fit.`,
    );
  }, [disposeSourceTextures]);

  const loadBuiltInMaterial = useCallback(async (key: string) => {
    const sample = MATERIALX_SAMPLES.find((s) => s.key === key);
    if (!sample) return;
    try {
      setStatus(`Loading ${sample.file}...`);
      const loader = new MaterialXLoader().setPath('/materialx/');
      const asset: any = await loader.loadAsync(sample.file, { uvSpace: 'top-left', throwOnErrors: true });
      const materials = asset?.materials ?? asset;
      const material = Object.values(materials).find(isPhysicalNodeMaterial) ?? Object.values(materials)[0];
      if (!material) throw new Error('MaterialXLoader did not produce any materials.');
      if (asset.texturesReady) await asset.texturesReady;
      setSourceMaterial(material, sample.file);
    } catch (err) {
      console.error(err);
      setStatus(errorMessage(err));
    }
  }, [setSourceMaterial]);

  useEffect(() => {
    void loadBuiltInMaterial(DEFAULT_MATERIALX_KEY);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadMtlxFile = useCallback(async (file: File) => {
    try {
      setStatus(`Loading ${file.name}...`);
      const loader = new MaterialXLoader();
      const asset: any = loader.parseBuffer(await file.arrayBuffer(), file.name, { uvSpace: 'top-left', throwOnErrors: true });
      const materials = asset?.materials ?? asset;
      const material = Object.values(materials).find(isPhysicalNodeMaterial) ?? Object.values(materials)[0];
      if (!material) throw new Error('MaterialXLoader did not produce any materials.');
      if (asset.texturesReady) await asset.texturesReady;
      setBuiltInKey('');
      setSourceMaterial(material, file.name);
    } catch (err) {
      console.error(err);
      setStatus(errorMessage(err));
    }
  }, [setSourceMaterial]);

  const onMtlxFileSelected = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) await loadMtlxFile(file);
  }, [loadMtlxFile]);

  const train = useCallback(async () => {
    const material = materialXMaterialRef.current;
    if (!material || !channelClassification) return;

    if (channelClassification.activeChannels.length === 0) {
      setStatus('Every channel is constant on this material - nothing to train.');
      return;
    }

    setIsTraining(true);
    resetLoss();
    setStatus('Baking material channels...');

    try {
      const renderer = await getSharedRenderer();
      disposeSourceTextures();
      const renderTargets = await bakeMaterialToTextures(
        renderer,
        material,
        Number(values.bakeResolution),
        channelClassification.activeChannels,
        uvTransformRef.current,
      );
      sourceTexturesRef.current = renderTargets;

      setStatus('Training...');

      const trainer = new NTCTrainer({
        channels: 4,
        levels: Number(values.levels),
        baseResolution: Number(values.baseResolution),
        hiddenSizes: [Number(values.hiddenSize), Number(values.hiddenSize)],
        hiddenActivation: values.hiddenActivation,
        outputChannels: channelClassification.totalChannels,
        channelActivations: buildChannelActivations(channelClassification.activeChannels) as string[],
        batchSize: Number(values.batchSize),
        iterations: Number(values.iterations),
        learningRate: Number(values.learningRate),
        // Only `mode` is meaningful here (matches the upstream three.js
        // example, which passes exactly this) - NTCTrainer.resolveQuantizationConfig
        // fills in the rest (target/range/perLevel) from its own defaults at
        // runtime; the TS port's option type is stricter than that runtime
        // behavior, so this is cast rather than hand-duplicating those defaults.
        quantization: { mode: values.quantization } as any,
        uvTransform: uvTransformRef.current,
        seed: 1,
      });
      activeTrainerRef.current = trainer;

      const result = await trainer.train({
        renderer,
        sourceTextures: renderTargets.map((rt: any) => rt.texture),
        onProgress: (progress: any) => {
          const isLast = progress.iteration >= progress.iterations - 1;
          addLossPoint({ iteration: progress.iteration, loss: progress.loss }, isLast);
          rebuildPreviewMaterial(renderer, progress.cpuModel, channelClassification, Number(values.lodBias));
        },
      });

      rebuildPreviewMaterial(renderer, result.cpuModel, channelClassification, Number(values.lodBias));
      setStatus(result.stoppedEarly ? 'Stopped.' : 'Training complete.');
    } catch (err) {
      console.error(err);
      setStatus(errorMessage(err));
    } finally {
      activeTrainerRef.current = null;
      setIsTraining(false);
    }
  }, [channelClassification, values, disposeSourceTextures, rebuildPreviewMaterial, resetLoss, addLossPoint]);

  const stopTraining = useCallback(() => {
    if (!activeTrainerRef.current) return;
    activeTrainerRef.current.abort();
    setStatus('Stopping...');
  }, []);

  const saveNtc = useCallback(() => {
    const material = previewMaterialRef.current;
    if (!material || !material.cpuModel) return;
    const name = sourceName || 'Untitled neural material';
    const manifest = new NTCExporter().parse(material, { name, source: 'Exported from the three-ntc trainer.' });
    const safeName = name.trim().replace(/[^a-z0-9-_]+/gi, '_') || 'neural-material';
    download(`${safeName}.ntc`, JSON.stringify(manifest));
    setStatus(`Saved ${safeName}.ntc.`);
  }, [sourceName]);

  const modelSizeSummary = useMemo(() => {
    const outputChannels = channelClassification ? channelClassification.totalChannels : MAX_TOTAL_CHANNELS;
    const channels = 4;
    const resolutions = computeGridLevels(Number(values.baseResolution), Number(values.levels));
    const latentTexels = computeGridLatentTexels(resolutions);
    const gridParams = latentTexels * channels;
    const inputSize = channels + 1;
    const mlpSpec = { inputSize, hiddenSize: Number(values.hiddenSize), hiddenLayers: 2, outputSize: outputChannels };
    const mlpParams = computeMLPParamCount(mlpSpec);
    const flops = computeMLPFlops(mlpSpec);
    return formatModelSizeSummary(computeModelFootprint({ gridParams, mlpParams, flops }));
  }, [channelClassification, values.baseResolution, values.levels, values.hiddenSize]);

  const applyPreset = useCallback((name: string) => {
    form.setFieldValue('preset', name);
    const profile = getNTCProfile(name);
    if (!profile) return;
    form.setFieldValue('levels', profile.levels);
    form.setFieldValue('baseResolution', profile.baseResolution);
    form.setFieldValue('hiddenSize', profile.hiddenSizes[profile.hiddenSizes.length - 1]);
    form.setFieldValue('hiddenActivation', profile.hiddenActivation);
  }, [form]);

  return (
    <div className="relative flex flex-1 flex-col gap-4 p-4">
      {/* Settings panel: normal-flow block on mobile (stacked above the
          viewer); becomes an absolutely-positioned, collapsible overlay on
          top of the viewer at lg+ - `left-0/top-0/bottom-0` line up with the
          padding edge the container's own `p-4` already provides, so no
          doubled-up inset math is needed. */}
      <div
        className={cn(
          'flex flex-col gap-4 overflow-y-auto',
          'lg:absolute lg:top-0 lg:bottom-0 lg:left-0 lg:z-10 lg:rounded-xl lg:border lg:border-border lg:bg-background/95 lg:p-4 lg:shadow-lg lg:backdrop-blur-sm',
          settingsOpen ? 'lg:w-[380px]' : 'lg:w-auto',
        )}
      >
        <div className="flex items-center justify-between gap-2">
          <div className={cn(!settingsOpen && 'lg:hidden')}>
            <h1 className="text-lg font-semibold">MaterialX Trainer</h1>
            <p className="text-sm text-muted-foreground">
              Fit a MaterialX material into a neural texture compression (.ntc) model, right in the browser.
            </p>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="hidden shrink-0 lg:inline-flex lg:w-8 lg:px-0"
            onClick={() => setSettingsOpen((open) => !open)}
            aria-label={settingsOpen ? 'Collapse settings' : 'Expand settings'}
          >
            {settingsOpen ? <PanelLeftClose className="size-4" /> : <PanelLeftOpen className="size-4" />}
          </Button>
        </div>

        <div className={cn('flex flex-col gap-4', !settingsOpen && 'lg:hidden')}>
        <div className="flex flex-wrap gap-2">
          {isTraining ? (
            <Button type="button" variant="outline" onClick={stopTraining}>
              Stop training
            </Button>
          ) : (
            <Button
              type="button"
              disabled={!hasSource}
              onClick={() => {
                ga.event('trainer-train', { materialx_key: builtInKey || sourceName || 'unknown' });
                void train();
              }}
            >
              Train
            </Button>
          )}
          <Button type="button" variant="outline" disabled={!hasTrainedModel} onClick={saveNtc}>
            Save .ntc
          </Button>
        </div>

        <CollapsibleSection title="Source">
              <Field>
                <FieldLabel>Built-in MaterialX</FieldLabel>
                <Select
                  value={builtInKey}
                  onValueChange={(key) => {
                    setBuiltInKey(key);
                    ga.event('trainer-default-materialx', { materialx_key: key });
                    void loadBuiltInMaterial(key);
                  }}
                  disabled={isTraining}
                >
                  <SelectTrigger size="sm">
                    <SelectValue placeholder="Choose an example…" />
                  </SelectTrigger>
                  <SelectContent>
                    {MATERIALX_SAMPLES.map((sample) => (
                      <SelectItem key={sample.key} value={sample.key}>
                        {sample.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>

              <div
                className={`flex items-center gap-2 rounded-lg border-2 border-dashed p-2 transition-colors ${
                  mtlxDragOver ? 'border-primary bg-accent' : 'border-transparent'
                }`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setMtlxDragOver(true);
                }}
                onDragLeave={() => setMtlxDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setMtlxDragOver(false);
                  const file = e.dataTransfer.files[0];
                  if (!file) return;
                  ga.event('trainer-drop-materialx', { file_name: file.name });
                  void loadMtlxFile(file);
                }}
              >
                <Button type="button" variant="outline" size="sm" disabled={isTraining} onClick={() => mtlxInputRef.current?.click()}>
                  Open .mtlx
                </Button>
                <span className="truncate text-sm text-muted-foreground">
                  {sourceName ?? 'loading default'} — or drag & drop a .mtlx file here
                </span>
              </div>
              <input ref={mtlxInputRef} type="file" accept=".mtlx,.zip,.mtlx.zip" hidden onChange={onMtlxFileSelected} />

              <form.Field name="bakeResolution">
                {(field) => (
                  <SelectFormField
                    field={field}
                    label="Bake resolution"
                    options={BAKE_RESOLUTION_OPTIONS}
                    parse={Number}
                  />
                )}
              </form.Field>
        </CollapsibleSection>

        <CollapsibleSection title="Network (grid + MLP)">
              <Field>
                <FieldLabel>Preset</FieldLabel>
                <Select value={values.preset} onValueChange={applyPreset}>
                  <SelectTrigger size="sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {NTC_PROFILE_NAMES.map((name) => (
                      <SelectItem key={name} value={name}>
                        {NTC_PROFILES[name].label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>

              <form.Field name="levels">
                {(field) => <SelectFormField field={field} label="Feature levels" options={GRID_LEVELS_OPTIONS} parse={Number} />}
              </form.Field>
              <form.Field name="baseResolution">
                {(field) => (
                  <SelectFormField field={field} label="Finest grid res" options={GRID_BASE_RESOLUTION_OPTIONS} parse={Number} />
                )}
              </form.Field>
              <form.Field name="hiddenSize">
                {(field) => (
                  <SelectFormField
                    field={field}
                    label="MLP hidden width (x2 layers)"
                    options={MLP_HIDDEN_SIZE_OPTIONS}
                    parse={Number}
                  />
                )}
              </form.Field>
              <form.Field name="hiddenActivation">
                {(field) => <SelectFormField field={field} label="MLP hidden activation" options={MLP_ACTIVATION_OPTIONS} />}
              </form.Field>
        </CollapsibleSection>

        <CollapsibleSection title="Training">
              <form.Field name="batchSize">
                {(field) => <SelectFormField field={field} label="Batch size" options={BATCH_SIZE_OPTIONS} parse={Number} />}
              </form.Field>

              <form.Field name="iterations">
                {(field) => (
                  <Field>
                    <FieldLabel>Iterations: {field.state.value}</FieldLabel>
                    <Slider
                      min={200}
                      max={20000}
                      step={100}
                      value={[field.state.value]}
                      onValueChange={([v]) => field.handleChange(v)}
                    />
                  </Field>
                )}
              </form.Field>

              <form.Field name="learningRate">
                {(field) => (
                  <Field>
                    <FieldLabel>Learning rate: {field.state.value.toFixed(3)}</FieldLabel>
                    <Slider
                      min={0.001}
                      max={0.05}
                      step={0.001}
                      value={[field.state.value]}
                      onValueChange={([v]) => field.handleChange(v)}
                    />
                  </Field>
                )}
              </form.Field>

              <Field>
                <FieldLabel>Quantization</FieldLabel>
                <Select
                  value={values.quantization}
                  onValueChange={(v) => form.setFieldValue('quantization', v as 'none' | 'uint8')}
                >
                  <SelectTrigger size="sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {QUANTIZATION_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
        </CollapsibleSection>

        <CollapsibleSection title="Object">
              <form.Field name="shape">
                {(field) => <SelectFormField field={field} label="Shape" options={SHAPE_OPTIONS} />}
              </form.Field>
        </CollapsibleSection>

        <CollapsibleSection title="View">
              <form.Field name="interpolation">
                {(field) => (
                  <Field orientation="horizontal">
                    <FieldLabel htmlFor={field.name}>Interpolation (feature grid)</FieldLabel>
                    <Switch
                      id={field.name}
                      checked={field.state.value}
                      onCheckedChange={(checked) => {
                        field.handleChange(checked);
                        previewMaterialRef.current?.setInterpolation?.(checked);
                      }}
                    />
                  </Field>
                )}
              </form.Field>

              <form.Field name="lodBias">
                {(field) => (
                  <Field>
                    <FieldLabel>LOD bias (force finer): {field.state.value.toFixed(2)}</FieldLabel>
                    <Slider
                      min={-4}
                      max={16}
                      step={0.25}
                      value={[field.state.value]}
                      onValueChange={([v]) => field.handleChange(v)}
                    />
                  </Field>
                )}
              </form.Field>
        </CollapsibleSection>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-line text-sm text-muted-foreground">{status}</p>
          </CardContent>
        </Card>
        </div>
      </div>

      <div className="flex min-h-[320px] flex-1 flex-col gap-4">
        <div className="relative min-h-[320px] flex-1 overflow-hidden rounded-xl border border-border bg-black">
          <NTCViewer material={previewMaterial} teacherMaterial={teacherMaterial} shape={values.shape} />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-[220px_1fr]">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Model size</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-muted-foreground">{modelSizeSummary}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Training loss</CardTitle>
                <span className="text-xs text-muted-foreground">{lossIps}</span>
              </div>
            </CardHeader>
            <CardContent>
              {lossPoints.length > 1 ? (
                <Chart definition={lossChartDefinition} height={160} ariaLabel="Training loss, log scale" />
              ) : (
                <p className="flex h-[160px] items-center justify-center text-sm text-muted-foreground">
                  No training data yet.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
