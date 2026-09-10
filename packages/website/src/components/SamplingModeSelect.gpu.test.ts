import { createElement, useState } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { NTCLoader, NTCNodeMaterial, type NTCSamplingMode } from "three-ntc";
import { SamplingModeSelect } from "./SamplingModeSelect.js";
import brick from "../../public/ntc/brick.ntc?raw";

it("offers all runtime modes, starts at nearest, and applies changes to the material", async () => {
  const { cpuModel, channelClassification } = new NTCLoader().parse(brick);
  const material = new NTCNodeMaterial(cpuModel, channelClassification);
  function Harness() {
    const [mode, setMode] = useState<NTCSamplingMode>(material.samplingMode);
    return createElement(SamplingModeSelect, {
      value: mode,
      onChange: (next) => {
        material.setSamplingMode(next);
        setMode(next);
      },
    });
  }
  const fixture = document.createElement("div");
  document.body.append(fixture);
  const root = createRoot(fixture);
  try {
    root.render(createElement(Harness));
    await expect.poll(() => fixture.querySelector("select")?.value).toBe("nearest");
    const select = fixture.querySelector("select")!;
    expect([...select.options].map((option) => option.value)).toEqual([
      "nearest",
      "stochastic",
      "trilinear",
    ]);
    for (const mode of ["stochastic", "trilinear", "nearest"] as const) {
      select.value = mode;
      select.dispatchEvent(new Event("change", { bubbles: true }));
      await expect.poll(() => material.samplingMode).toBe(mode);
      await expect.poll(() => select.value).toBe(mode);
      if (mode === "stochastic")
        await expect
          .poll(() => fixture.textContent)
          .toContain("Noisy until temporal reconstruction");
    }
  } finally {
    root.unmount();
    fixture.remove();
    material.dispose();
  }
});
