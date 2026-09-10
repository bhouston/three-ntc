import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

// Run serially: concurrent browser profiles would compete for the same GPU.
const browser = process.argv[2] || "chromium";
if (!["chromium", "webkit", "safari"].includes(browser)) {
  throw new Error("Usage: pnpm profile:browser [chromium|webkit|safari] [output-directory]");
}
const output = resolve(process.argv[3] || `docs/metrics/browser-profile-${browser}-${Date.now()}`);
await mkdir(output, { recursive: true });
const profiles = [
  "packages/website/src/components/NTCViewer.gpu.test.ts",
  "packages/three-ntc-trainer/src/NTCPreviewProfile.gpu.test.ts",
  "packages/three-ntc/src/NTCBrickProfile.gpu.test.ts",
];
// Native Safari uses the display's actual DPR; Playwright can emulate DPR 1 and 2.
const scales = browser === "safari" ? [null] : [1, 2];
const results = [];
for (const scale of scales) {
  const name = scale === null ? "native-display" : `dpr-${scale}`;
  const env = { ...process.env, NTC_BROWSER: browser };
  if (scale !== null) env.NTC_DEVICE_SCALE_FACTOR = String(scale);
  if (process.platform === "darwin" && !env.NTC_GPU_BACKEND) env.NTC_GPU_BACKEND = "metal";
  const args = [
    "exec",
    "vitest",
    "run",
    "--project",
    "gpu",
    "--fileParallelism=false",
    ...profiles,
  ];
  const startedAt = new Date().toISOString();
  let log = "";
  const child = spawn("pnpm", args, { env, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", (chunk) => {
    log += chunk;
    process.stdout.write(chunk);
  });
  child.stderr.on("data", (chunk) => {
    log += chunk;
    process.stderr.write(chunk);
  });
  const exitCode = await new Promise((accept, reject) => {
    child.once("error", reject);
    child.once("close", (code) => accept(code ?? 1));
  });
  await writeFile(resolve(output, `${name}.log`), log);
  const metrics = log
    .split("\n")
    .filter((line) => line.startsWith("NTC_METRIC "))
    .map((line) => JSON.parse(line.slice("NTC_METRIC ".length)));
  results.push({ browser, requestedDpr: scale, startedAt, exitCode, metrics });
  // Preserve failures and their logs; an empty set of measurements is not a pass.
  await writeFile(resolve(output, "results.json"), JSON.stringify(results, null, 2) + "\n");
  if (exitCode !== 0 || metrics.length === 0) process.exitCode = 1;
}
console.log(`Profile results: ${output}`);
