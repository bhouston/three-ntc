import { appendFileSync, mkdirSync } from 'node:fs';
import { spawn, execFileSync } from 'node:child_process';
import { mkdir, writeFile, copyFile, mkdtemp, symlink } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const label = process.argv[2];
if (!label || !/^[a-z0-9-]+$/.test(label)) throw new Error('Supply a lowercase measurement label');
const root = fileURLToPath(new URL('../', import.meta.url));
let cwd = process.argv[3];
if (!cwd) {
  cwd = await mkdtemp(join(tmpdir(), 'ntc-quality-'+label+'-'));
  const files = execFileSync('git', ['ls-files','-c','-o','--exclude-standard'], {cwd:root,encoding:'utf8'}).trim().split('\n');
  for (const file of new Set(files)) {
    if (!/\.(ts|tsx|js|mjs|json|yaml|html|mtlx|ntc)$/.test(file) || file.startsWith('docs/')) continue;
    await mkdir(dirname(join(cwd,file)),{recursive:true});
    await copyFile(join(root,file),join(cwd,file));
  }
  for (const sub of ['', 'packages/three-ntc', 'packages/three-ntc-trainer', 'packages/website']) {
    await mkdir(join(cwd,sub),{recursive:true});
    await symlink(join(root,sub,'node_modules'),join(cwd,sub,'node_modules'),'dir');
  }
}
console.log('Benchmark checkout: '+cwd);
mkdirSync(`${root}docs/metrics`, {recursive:true});
await writeFile(`${root}docs/metrics/${label}.log`, '');
const child = spawn(process.execPath, [fileURLToPath(new URL('vitest.mjs', import.meta.resolve('vitest/package.json'))),
  'run', '--project', 'gpu', '--no-file-parallelism',
  'packages/three-ntc-trainer/src/NTCQuality.gpu.test.ts'], {cwd, env:process.env});
let log = '';
for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
  log += chunk;
  appendFileSync(`${root}docs/metrics/${label}.log`, chunk);
  for (const line of String(chunk).split('\n')) if (line.includes('NTC_METRIC')) console.log(line);
});
const code = await new Promise(resolve => child.on('close', resolve));
const records = [...log.matchAll(/NTC_METRIC (\{[^\n]+\})/g)].map(m => JSON.parse(m[1]));
await mkdir(`${root}docs/metrics`, {recursive:true});
await writeFile(`${root}docs/metrics/${label}.log`, log);
if (code || records.length !== 6) throw new Error(`Benchmark failed (${code}, ${records.length} records); see docs/metrics/${label}.log`);
await writeFile(`${root}docs/metrics/${label}.json`, JSON.stringify(records,null,2)+'\n');
const mse = records.reduce((s,r) => s+r.exported.mse,0)/records.length;
console.log(JSON.stringify({label, mse, psnr:-10*Math.log10(mse)}));
