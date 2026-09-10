import { readFile, writeFile } from "node:fs/promises";
import ts from "typescript";
// Compile the production pure filter, without requiring a separate TS runner.
const source = await readFile(
  new URL("../packages/three-ntc-trainer/src/NTCMipFilter.ts", import.meta.url),
  "utf8",
);
const js = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext },
}).outputText;
const { createLanczosMipChain } = await import(
  "data:text/javascript;base64," + Buffer.from(js).toString("base64")
);
const records = [];
for (const frequency of [1 / 8, 3 / 8]) {
  const signal = (x) => Math.cos(2 * Math.PI * frequency * x);
  const data = Float32Array.from({ length: 64 * 64 }, (_, i) => signal(i % 64));
  const mip = createLanczosMipChain(data, 64, 64, 1)[1];
  let boxMse = 0,
    lanczosMse = 0;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++) {
      const ideal = frequency < 0.25 ? signal(2 * x + 0.5) : 0;
      boxMse += ((signal(2 * x) + signal(2 * x + 1)) / 2 - ideal) ** 2 / 1024;
      lanczosMse += (mip.data[y * 32 + x] - ideal) ** 2 / 1024;
    }
  records.push({ frequency, boxMse, lanczosMse });
}
await writeFile(
  new URL("../docs/metrics/source-filter.json", import.meta.url),
  JSON.stringify(records, null, 2) + "\n",
);
console.log(records);
