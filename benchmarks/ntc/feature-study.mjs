// A bounded, resumable feature study. GPU cases run sequentially.
import { spawnSync, spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const label=process.argv[2]??'feature-contributions-4000-v2';
if(!/^[\w-]+$/.test(label))throw new Error('Invalid study label');
const out=resolve(root,'output/benchmarks',label);
const python=process.env.NTC_ANALYSIS_PYTHON??'python3';
let stopping=false,activeChild;
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>{stopping=true;activeChild?.kill('SIGTERM');});
const variants=['reference','pe-none_g0-concatenate_uint4','pe-cell_g0-concatenate_uint4',
  'pe-tile_g0-bilinear_uint4','pe-none_g0-bilinear_uint4',
  'pe-tile_g0-concatenate_uint2','pe-tile_g0-concatenate_uint8',
  'activation-relu','hidden-16','hidden-64','relu-hidden-16','g1-off',
  'base-128','base-512','levels-2','grid-channels-8-12'];
mkdirSync(out,{recursive:true});
if(!existsSync(resolve(out,'plan.json'))) {
  const planned=spawnSync(process.execPath,['benchmarks/ntc/run.mjs','--label',label,'--materials','brick,uv_grid,wood',
    '--variants',variants.join(','),'--seeds','1,2,3','--plan-only'],{cwd:root,stdio:'inherit'});
  if(planned.status!==0)throw new Error('Could not create plan');
  const path=resolve(out,'plan.json'),plan=JSON.parse(readFileSync(path));
  // Retain the fixed random shuffle within each seed block. This yields a
  // complete exploratory screen before later replicates, without selecting winners.
  plan.cases.sort((a,b)=>a.seed-b.seed);
  plan.study={name:'controlled-feature-screen-v2',variants,materials:['brick','uv_grid','wood'],
    primary:'exported complete-chain range-normalized MSE / PSNR',
    scheduling:'Fixed randomized order within seed blocks; all variants receive all three seeds.',
    analysis:'Paired seed effects conditional on reference; PE x G0 at uint4 and activation x width interactions; size-quality Pareto points.',
    sourceExclusion:'checkerboard_normal withheld because its prior reference was constant',
    fidelity:'G0 noise path corrected; this controlled study uses STE for both grids',
    stopping:'4000 iterations per result; one retry pass for failed cases; no early stopping or adaptive selection'};
  writeFileSync(path,JSON.stringify(plan,null,2));
}
function analyze() {
  const result=spawnSync(python,['benchmarks/ntc/feature-findings.py',out],{cwd:root,stdio:'inherit'});
  if(result.status!==0)console.error('Analysis failed; raw training results remain available.');
}
async function run(extra=[]) {
  const child=spawn(process.execPath,['benchmarks/ntc/run.mjs','--label',label,...extra],{cwd:root,stdio:['ignore','pipe','inherit']});
  activeChild=child;
  let buffered='',completed=0;
  child.stdout.on('data',data=>{
    process.stdout.write(data);buffered+=data;
    const lines=buffered.split('\n');buffered=lines.pop();
    for(const line of lines)if(line.startsWith('DONE ')&&++completed%3===0)analyze();
  });
  return await new Promise(resolve=>child.on('close',resolve));
}
analyze();
const status=await run();
if(status!==0){analyze();process.exit(Number(status)||1);}
const summary=JSON.parse(readFileSync(resolve(out,'summary.json')));
if(stopping||summary.pending){analyze();process.exit(0);}
if(summary.failed)await run(['--retry-failed']);
analyze();
writeFileSync(resolve(out,'study-finished.json'),JSON.stringify({finishedAt:new Date().toISOString(),...JSON.parse(readFileSync(resolve(out,'summary.json'))),records:undefined,effects:undefined,profileSummaries:undefined},null,2));

const final=JSON.parse(readFileSync(resolve(out,'summary.json')));
if(!final.pending) {
  const assets=resolve(root,'docs/benchmark-reports',label);mkdirSync(assets,{recursive:true});
  copyFileSync(resolve(out,'feature-effects.json'),resolve(assets,'feature-effects.json'));
  copyFileSync(resolve(out,'feature-effects.png'),resolve(assets,'feature-effects.png'));
  const markdown=readFileSync(resolve(out,'findings.md'),'utf8')
    .replace('(feature-effects.png)',`(benchmark-reports/${label}/feature-effects.png)`)
    .replace('`feature-effects.json`',`[feature-effects.json](benchmark-reports/${label}/feature-effects.json)`);
  writeFileSync(resolve(root,'docs',label+'-findings.md'),markdown);
}
