import { NTC_SAMPLING_MODES, type NTCSamplingMode } from "three-ntc";

const labels: Record<NTCSamplingMode, string> = {
  nearest: "Nearest neighbor",
  stochastic: "Stochastic",
  trilinear: "Trilinear",
};

export function SamplingModeSelect({
  value,
  onChange,
}: {
  value: NTCSamplingMode;
  onChange: (mode: NTCSamplingMode) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="flex flex-col gap-2 text-sm">
        Sampling
        <select
          className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
          value={value}
          onChange={(event) => onChange(event.target.value as NTCSamplingMode)}
        >
          {NTC_SAMPLING_MODES.map((mode) => (
            <option key={mode} value={mode}>
              {labels[mode]}
            </option>
          ))}
        </select>
      </label>
      <p className="text-xs text-muted-foreground">
        {value === "nearest"
          ? "One decoded texel per sample."
          : value === "stochastic"
            ? "One decoded texel per sample. Noisy until temporal reconstruction is added."
            : "Blends eight decoded texels per sample. More expensive."}
      </p>
    </div>
  );
}
