import { gzipSync, brotliCompressSync, constants as zlibConstants } from 'node:zlib';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, copyFile, symlink, stat } from 'node:fs/promises';
import { existsSync, createWriteStream } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeCases } from './design.mjs';
import { generateReport } from './report.mjs';

const root=fileURLToPath(new URL('../../',import.meta.url));
const args=process.argv.slice(2);
const arg=(name,fallback)=>{const i=args.indexOf('--'+name);return i<0?fallback:args[i+1];};
const label=arg('label','materialx-ablation');
if(!/^[a-zA-Z0-9_-]+$/.test(label))throw new Error('Invalid label');
const out=resolve(root,'output/benchmarks',label),snapshot=join(out,'snapshot');
await mkdir(join(out,'results'),{recursive:true});await mkdir(join(out,'logs'),{recursive:true});
const planPath=join(out,'plan.json');
const dependencyVersions={};
for(const name of ['three','vitest','vite','playwright']) dependencyVersions[name]=JSON.parse(await readFile(join(root,'node_modules',name,'package.json'),'utf8')).version;
const environment={node:process.version,platform:process.platform,arch:process.arch,dependencyVersions,
  ...(process.platform==='win32'?{browserChannel:process.env.NTC_BROWSER_CHANNEL??'chromium',headed:process.env.NTC_GPU_HEADED==='1'}:{})};
let plan;
if(existsSync(planPath)) {
  plan=JSON.parse(await readFile(planPath,'utf8'));
  if(JSON.stringify(plan.environment)!==JSON.stringify(environment))throw new Error('Runtime/dependency versions changed; choose a new label');
  if(args.includes('--fresh'))throw new Error('Choose a new label for a fresh run');
  // No new cases/options can silently enter an existing snapshot.
  for(const key of ['iterations','resolution','seeds','suite','materials','variants'])if(args.includes('--'+key))throw new Error('Resume using --label only; use a new label to change the plan');
} else {
  const iterations=Number(arg('iterations','4000')),resolution=Number(arg('resolution','1024'));
  const seeds=arg('seeds','1,2,3').split(',').map(Number);
  if(!Number.isInteger(iterations)||iterations<1||!Number.isInteger(Math.log2(resolution))||resolution<16||seeds.some(s=>!Number.isInteger(s))||new Set(seeds).size!==seeds.length)throw new Error('Invalid design settings');
  const suite=arg('suite','all');if(!['all','controlled','native'].includes(suite))throw new Error('Invalid suite');
  const cases=makeCases({seeds,iterations,resolution,suite,
    ...(arg('materials')?{onlyMaterials:arg('materials').split(',')}:{}),
    ...(arg('variants')?{onlyVariants:arg('variants').split(',')}:{}),
  });
  if(!cases.length)throw new Error('Empty design');
  await mkdir(snapshot,{recursive:true});
  const exportedSourcePath=join(root,'benchmark-source.json');
  const exportedSource=existsSync(exportedSourcePath)?JSON.parse(await readFile(exportedSourcePath,'utf8')):null;
  const files=exportedSource ? exportedSource.files : [...new Set(execFileSync('git',['ls-files','-c','-o','--exclude-standard','-z'],{cwd:root}).toString().split('\0').filter(Boolean))].sort();
  const hash=createHash('sha256');const hashes={};
  for(const file of files) {
    if(file.startsWith('output/')||file.startsWith('docs/')||file.includes('/dist/')||file.includes('node_modules/'))continue;
    const source=join(root,file);if(!(await stat(source)).isFile())continue;
    const bytes=await readFile(source),digest=createHash('sha256').update(bytes).digest('hex');
    hashes[file]=digest;hash.update(file+'\0'+digest+'\0');
    await mkdir(dirname(join(snapshot,file)),{recursive:true});await copyFile(source,join(snapshot,file));
  }
  for(const sub of ['', 'packages/three-ntc','packages/three-ntc-trainer','packages/website']) {
    await mkdir(join(snapshot,sub),{recursive:true});await symlink(join(root,sub,'node_modules'),join(snapshot,sub,'node_modules'),process.platform==='win32'?'junction':'dir');
  }
  plan={schemaVersion:1,createdAt:new Date().toISOString(),commit:exportedSource?.commit ?? execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
    sourceHash:hash.digest('hex'),fileHashes:hashes,environment,seeds,iterations,resolution,suite,
    benchmark:iterations===4000&&resolution===1024?'production':'smoke',
    backend:arg('backend',process.platform==='darwin'?'metal':'native'),cases};
  await writeFile(planPath,JSON.stringify(plan,null,2));
}
console.log(`Plan: ${plan.cases.length} cases × ${plan.iterations} iterations; ${plan.benchmark}; source ${plan.sourceHash.slice(0,12)}`);
console.log(`Results: ${out}`);
if(args.includes('--plan-only'))process.exit(0);
if(args.includes('--report-only')) {await generateReport(out);process.exit(0);}
// An atomic directory prevents two resumed drivers from training the same cases.
const lock=join(out,'runner.lock');
try {await mkdir(lock);}catch{throw new Error(`Runner lock exists: ${lock}. Check its owner before removing a stale lock.`);}
await writeFile(join(lock,'owner.json'),JSON.stringify({pid:process.pid,startedAt:new Date().toISOString()}));
let stopping=false;const stop=()=>{stopping=true;};process.on('SIGINT',stop);process.on('SIGTERM',stop);
const limit=Number(arg('limit','Infinity'));let completed=0;
try {
  for(const testCase of plan.cases) {
    if(stopping||completed>=limit)break;
    const resultPath=join(out,'results',testCase.id+'.json');
    if(existsSync(resultPath)) {
      const prior=JSON.parse(await readFile(resultPath,'utf8'));
      if(prior.status==='complete' || !args.includes('--retry-failed'))continue;
    }
    const candidatePath=resultPath+'.candidate';
    const casePath=join(out,'current-case.json');await writeFile(casePath,JSON.stringify(testCase));
    const log=createWriteStream(join(out,'logs',testCase.id+'.log'),{flags:'a'});
    console.log(`START ${testCase.id}`);
    const started=Date.now();
    const child=spawn(process.execPath,[fileURLToPath(new URL('vitest.mjs',import.meta.resolve('vitest/package.json'))),
      'run','--project','gpu','--no-file-parallelism','benchmarks/ntc/run-case.gpu.test.ts'],{
      cwd:snapshot,env:{...process.env,NTC_GPU_BACKEND:plan.backend,NTC_BENCHMARK_CASE:casePath,NTC_BENCHMARK_RESULT:candidatePath},stdio:['ignore','pipe','pipe'],
    });
    for(const stream of [child.stdout,child.stderr])stream.on('data',data=>{log.write(data);if(String(data).includes('NTC_PROGRESS'))process.stdout.write(data);});
    const exitCode=await new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',resolve);});log.end();
    if(exitCode!==0 || !existsSync(candidatePath)) {
      await writeFile(resultPath,JSON.stringify({...testCase,status:'failed',exitCode,elapsedMs:Date.now()-started,log:`logs/${testCase.id}.log`}));
      console.error(`FAILED ${testCase.id}; see log`);
    } else {
      const record=JSON.parse(await readFile(candidatePath,'utf8'));record.sourceHash=plan.sourceHash;
      if(record.completedIterations!==plan.iterations)throw new Error('Incomplete result was labeled complete');
      if(record.manifest) {
        const bytes=Buffer.from(JSON.stringify(record.manifest));
        const gzip=gzipSync(bytes,{level:9});
        const brotli=brotliCompressSync(bytes,{params:{[zlibConstants.BROTLI_PARAM_QUALITY]:5}});
        await mkdir(join(out,'models'),{recursive:true});
        for(const [suffix,data] of [['.ntc',bytes],['.ntc.gz',gzip],['.ntc.br',brotli]])
          await writeFile(join(out,'models',testCase.id+suffix),data);
        record.storage.gzipBytes=gzip.length;record.storage.brotliBytes=brotli.length;
        record.storage.transportCompression={gzipLevel:9,brotliQuality:5};
        record.modelPath='models/'+testCase.id+'.ntc';delete record.manifest;
      }
      if(record.preview?.dataUrl) {
        await mkdir(join(out,'previews'),{recursive:true});
        await writeFile(join(out,'previews',testCase.id+'.png'),Buffer.from(record.preview.dataUrl.split(',')[1],'base64'));
        record.preview={channel:record.preview.channel,path:`previews/${testCase.id}.png`};
      }
      await writeFile(resultPath+'.tmp',JSON.stringify(record));
      const {rename}=await import('node:fs/promises');await rename(resultPath+'.tmp',resultPath);
      const {rm}=await import('node:fs/promises');await rm(candidatePath,{force:true});await rm(candidatePath+'.progress.json',{force:true});
      console.log(`DONE ${testCase.id}: ${record.quality.exported.psnr.toFixed(3)} dB, ${(record.trainingMs/1000).toFixed(1)}s`);
    }
    completed++;await generateReport(out);
  }
  await generateReport(out);
} finally {const {rm}=await import('node:fs/promises');await rm(lock,{recursive:true,force:true});}
