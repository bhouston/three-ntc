import { build } from 'esbuild';
import { gzipSync } from 'node:zlib';
import { appendFileSync } from 'node:fs';

// Measure complete package exports with three.js supplied by the consumer.
// Bundle workspace runtime code into the trainer measurement.
const limits = { 'three-ntc': 11000, 'three-ntc-trainer': 52000 };
const rows = ['| Package | Minified gzip | Limit |', '| --- | ---: | ---: |'];
for (const [name, limit] of Object.entries(limits)) {
  const result = await build({
    entryPoints: [`packages/${name}/dist/index.js`],
    bundle: true,
    minify: true,
    write: false,
    format: 'esm',
    platform: 'browser',
    external: ['three', 'three/*'],
  });
  const bytes = gzipSync(result.outputFiles[0].contents).byteLength;
  rows.push(`| ${name} | ${bytes} B | ${limit} B |`);
  if (bytes > limit) process.exitCode = 1;
}
const summary = `${rows.join('\n')}\n`;
console.log(summary);
if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
