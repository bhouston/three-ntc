import { createFileRoute } from '@tanstack/react-router';
import { useForm, useStore } from '@tanstack/react-form';
import { type ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGoogleAnalytics } from 'tanstack-router-ga4';
import * as THREE from 'three';
import { defineChart, lineY } from '@tanstack/charts';
import { scaleLinear } from '@tanstack/charts/scales/linear';
import { Chart } from '@tanstack/charts/react';
import {
  bakeMaterialToTextures,
  classifyMaterialChannels,
  getMaterialXSampleUrl,
  GRID_BASE_RESOLUTION_OPTIONS,
  GRID_LEVELS_OPTIONS,
  getNTCProfileControls,
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
import { buildChannelActivations, MAX_TOTAL_CHANNELS, NTCNodeMaterial, type NTCSamplingMode } from 'three-ntc';

import { SamplingModeSelect } from '@/components/SamplingModeSelect';
import { ModelSizeSummary } from '@/components/ModelSizeSummary';
import { NTCViewer, type NTCViewerShape } from '@/components/NTCViewer';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldError, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { DEFAULT_LOD_BIAS } from '@/lib/ntc-examples';
import { getSharedRenderer } from '@/lib/renderer';
import { seoMeta } from '@/lib/seo';

export interface TrainerSearch {
  src?: string;
}

export const Route = createFileRoute('/trainer')({
  component: TrainerPage,
  validateSearch: (search: Record<string, unknown>): TrainerSearch => ({
    src: typeof search.src === 'string' ? search.src : undefined,
  }),
  head: () => ({
    meta: seoMeta({
      title: 'Three-NTC Trainer',
      description: 'Fit a MaterialX material into a neural texture compression model in the browser and export it as .ntc.',
      path: '/trainer',
    }),
  }),
});

const BAKE_RESOLUTION_OPTIONS = [128, 256, 512, 1024, 2048, 4096];
const BATCH_SIZE_OPTIONS = [1024, 2048, 4096, 8192, 16384];
const QUANTIZATION_OPTIONS: Array<{ value: 'none' | 'uint8' | 'uint4' | 'uint2'; label: string }> = [
  { value: 'none', label: 'Off' },
  { value: 'uint8', label: '8-bit' },
  { value: 'uint4', label: '4-bit' },
  { value: 'uint2', label: '2-bit' },
];
const SHAPE_OPTIONS: NTCViewerShape[] = ['torus', 'sphere', 'plane'];
const DEFAULT_MATERIALX_KEY = 'brick';

type FormValues = {
  bakeResolution: number;
  preset: string;
  levels: number;
  baseResolution: number;
  hiddenSize: number;
  hiddenLayers: number;
  hiddenActivation: string;
  positionalEncoding: boolean;
  dualGrid: boolean;
  batchSize: number;
  iterations: number;
  learningRate: number;
  quantization: 'none' | 'uint8' | 'uint4' | 'uint2';
  shape: NTCViewerShape;
  samplingMode: NTCSamplingMode;
  lodBias: number;
};

const DEFAULT_VALUES: FormValues = {
  bakeResolution: 1024,
  preset: 'mobile-fast',
  ...getNTCProfileControls('mobile-fast')!,
  positionalEncoding: true,
  dualGrid: true,
  batchSize: 8192,
  iterations: 10000,
  learningRate: 0.01,
  quantization: 'uint2',
  shape: 'torus',
  samplingMode: 'nearest',
  lodBias: DEFAULT_LOD_BIAS,
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
    <Field orientation="horizontal" data-invalid={field.state.meta.errors.length > 0}>
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

// A collapsible settings group - no card/box, just a bold header you can
// toggle. Native <details> gives us open/close state and a11y for free.
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="py-1.5 first:pt-0" open>
      <summary className="cursor-pointer text-sm font-bold text-foreground select-none">{title}</summary>
      <FieldGroup className="mt-2 gap-2">{children}</FieldGroup>
    </details>
  );
}

function TrainerPage() {
  const ga = useGoogleAnalytics();
  const navigate = Route.useNavigate();
  const { src } = Route.useSearch();
  const form = useForm({ defaultValues: DEFAULT_VALUES });
  const values = useStore(form.store, (state) => state.values);

  const [status, setStatus] = useState('Ready.');
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
    if (previous?.cpuModel === cpuModel) {
      previous.updateFromModel(cpuModel);
      return previous;
    }
    const material = new NTCNodeMaterial(cpuModel, classification, {
      renderer,
      // lodBias is wrapped in a live uniform node by NTCNodeMaterial itself -
      // this initial value only seeds it; further changes go through
      // material.setLodBias() (see the lodBias slider below) with no rebuild.
      lodBias,
      samplingMode: form.getFieldValue('samplingMode'),
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

  // Loads a MaterialX document from a URL - relative (matches a built-in
  // sample's own URL) or a full URL (external material shared via a link).
  // No `setPath`: MaterialXLoader resolves each referenced texture against
  // this URL's own directory, so both cases "just work" the same way.
  const loadMaterialXUrl = useCallback(async (url: string) => {
    try {
      setStatus(`Loading ${url}...`);
      const loader = new MaterialXLoader();
      const asset: any = await loader.loadAsync(url, { uvSpace: 'top-left', throwOnErrors: true });
      const materials = asset?.materials ?? asset;
      const material = Object.values(materials).find(isPhysicalNodeMaterial) ?? Object.values(materials)[0];
      if (!material) throw new Error('MaterialXLoader did not produce any materials.');
      if (asset.texturesReady) await asset.texturesReady;
      setSourceMaterial(material, url);
    } catch (err) {
      console.error(err);
      setStatus(errorMessage(err));
    }
  }, [setSourceMaterial]);

  const loadBuiltInMaterial = useCallback(async (key: string) => {
    const url = getMaterialXSampleUrl(key);
    if (!url) return;
    setBuiltInKey(key);
    await loadMaterialXUrl(url);
  }, [loadMaterialXUrl]);

  // ?src= drives the loaded material - a relative URL matches a built-in
  // sample by its own URL, a full URL loads external material shared via
  // that link. Falls back to the default built-in sample when absent.
  useEffect(() => {
    if (src) {
      const sample = MATERIALX_SAMPLES.find((s) => getMaterialXSampleUrl(s.key) === src);
      setBuiltInKey(sample?.key ?? '');
      void loadMaterialXUrl(src);
    } else {
      void loadBuiltInMaterial(DEFAULT_MATERIALX_KEY);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

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
      void navigate({ search: {} });
      setSourceMaterial(material, file.name);
    } catch (err) {
      console.error(err);
      setStatus(errorMessage(err));
    }
  }, [setSourceMaterial, navigate]);

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
        gridChannels: 4,
        levels: Number(values.levels),
        baseResolution: Number(values.baseResolution),
        hiddenSizes: Array(Number(values.hiddenLayers)).fill(Number(values.hiddenSize)),
        hiddenActivation: values.hiddenActivation,
        positionalEncoding: values.positionalEncoding,
        dualGrid: values.dualGrid,
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

  const applyPreset = useCallback((name: string) => {
    form.setFieldValue('preset', name);
    const profile = getNTCProfileControls(name);
    if (!profile) return;
    form.setFieldValue('levels', profile.levels);
    form.setFieldValue('baseResolution', profile.baseResolution);
    form.setFieldValue('hiddenSize', profile.hiddenSize);
    form.setFieldValue('hiddenLayers', profile.hiddenLayers);
    form.setFieldValue('hiddenActivation', profile.hiddenActivation);
  }, [form]);

  return (
    // 2-column grid at lg+ (settings | main); settings spans both content
    // rows there so it sits to the left of the viewer and above the loss
    // graph at once. Single column on mobile, where the `order-*` classes
    // below give the requested viewer -> settings -> details stacking.
    <div className="grid flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[380px_1fr] lg:grid-rows-[minmax(320px,1fr)_auto]">
      <div className="order-2 flex flex-col gap-2 overflow-y-auto lg:order-none lg:col-start-1 lg:row-span-2 lg:row-start-1">
        <div>
          <h1 className="text-lg font-semibold">MaterialX Trainer</h1>
          <p className="text-sm text-muted-foreground">
            Fit a MaterialX material into a neural texture compression (.ntc) model, right in the browser.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {isTraining ? (
            <Button type="button" variant="destructive" onClick={stopTraining}>
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

        <fieldset disabled={isTraining} className="flex min-w-0 flex-col gap-2">
        <Section title="Source">
              <Field orientation="horizontal">
                <FieldLabel>Built-in MaterialX</FieldLabel>
                <Select
                  value={builtInKey}
                  onValueChange={(key) => {
                    ga.event('trainer-default-materialx', { materialx_key: key });
                    void navigate({ search: { src: getMaterialXSampleUrl(key) } });
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
                className={`flex items-center gap-2 border-2 border-dashed p-2 transition-colors ${
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
                  if (isTraining) return;
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
        </Section>

        <Section title="Network (grid + MLP)">
              <Field orientation="horizontal">
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
                    label="MLP hidden width"
                    options={MLP_HIDDEN_SIZE_OPTIONS}
                    parse={Number}
                  />
                )}
              </form.Field>
              <form.Field name="hiddenLayers">
                {(field) => <SelectFormField field={field} label="MLP hidden layers" options={[1,2]} parse={Number} />}
              </form.Field>
              <form.Field name="hiddenActivation">
                {(field) => <SelectFormField field={field} label="MLP hidden activation" options={MLP_ACTIVATION_OPTIONS} />}
              </form.Field>
              <form.Field name="positionalEncoding">
                {(field) => (
                  <Field orientation="horizontal">
                    <FieldLabel htmlFor={field.name}>Positional encoding</FieldLabel>
                    <Switch id={field.name} checked={field.state.value} onCheckedChange={(checked) => field.handleChange(checked)} />
                  </Field>
                )}
              </form.Field>
              <form.Field name="dualGrid">
                {(field) => (
                  <Field orientation="horizontal">
                    <FieldLabel htmlFor={field.name}>Dual grid G0/G1</FieldLabel>
                    <Switch id={field.name} checked={field.state.value} onCheckedChange={(checked) => field.handleChange(checked)} />
                  </Field>
                )}
              </form.Field>
        </Section>

        <Section title="Training">
              <form.Field name="batchSize">
                {(field) => <SelectFormField field={field} label="Batch size" options={BATCH_SIZE_OPTIONS} parse={Number} />}
              </form.Field>

              <form.Field name="iterations">
                {(field) => (
                  <Field orientation="horizontal">
                    <FieldLabel>Iterations: {field.state.value}</FieldLabel>
                    <Slider
                      disabled={isTraining}
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
                  <Field orientation="horizontal">
                    <FieldLabel>Learning rate: {field.state.value.toFixed(3)}</FieldLabel>
                    <Slider
                      disabled={isTraining}
                      min={0.001}
                      max={0.05}
                      step={0.001}
                      value={[field.state.value]}
                      onValueChange={([v]) => field.handleChange(v)}
                    />
                  </Field>
                )}
              </form.Field>

              <Field orientation="horizontal">
                <FieldLabel>Quantization</FieldLabel>
                <Select
                  value={values.quantization}
                  onValueChange={(v) => form.setFieldValue('quantization', v as 'none' | 'uint8' | 'uint4' | 'uint2')}
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
        </Section>

        </fieldset>

        <Section title="Object">
              <form.Field name="shape">
                {(field) => <SelectFormField field={field} label="Shape" options={SHAPE_OPTIONS} />}
              </form.Field>
        </Section>

        <Section title="View">
              <SamplingModeSelect value={values.samplingMode} onChange={mode => {
                form.setFieldValue('samplingMode', mode);
                previewMaterialRef.current?.setSamplingMode(mode);
              }} />

              <form.Field name="lodBias">
                {(field) => (
                  <Field orientation="horizontal">
                    <FieldLabel>LOD bias (force finer): {field.state.value.toFixed(2)}</FieldLabel>
                    <Slider
                      min={-4}
                      max={16}
                      step={0.25}
                      value={[field.state.value]}
                      onValueChange={([v]) => {
                        field.handleChange(v);
                        previewMaterialRef.current?.setLodBias?.(v);
                      }}
                    />
                  </Field>
                )}
              </form.Field>
        </Section>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-line text-sm text-muted-foreground">{status}</p>
          </CardContent>
        </Card>
      </div>

      <div className="order-1 min-h-[320px] max-h-[60cqw] overflow-hidden border border-border bg-black [container-type:inline-size] lg:col-start-2 lg:row-start-1">
        <NTCViewer material={previewMaterial} teacherMaterial={teacherMaterial} shape={values.shape} cameraDistance={2.2} />
      </div>

      <div className="order-3 grid grid-cols-1 gap-4 sm:grid-cols-[220px_1fr] lg:col-start-2 lg:row-start-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Model Info</CardTitle>
            </CardHeader>
            <CardContent>
              <ModelSizeSummary name={sourceName || 'Untitled'} classification={channelClassification} settings={values} outputChannels={channelClassification?.totalChannels ?? MAX_TOTAL_CHANNELS} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">Training loss</CardTitle>
                <span className="text-xs font-mono tabular-nums">
                  {lossPoints.length > 0 ? lossPoints[lossPoints.length - 1].loss.toExponential(3) : ''}
                </span>
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
  );
}
