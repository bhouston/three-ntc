import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeCases, variants, materials } from './design.mjs';
import { measure, summarizeMips, pairedStats, normalError } from './metrics.mjs';
test('production design crosses all three bits, both samplers and three encodings with paired seeds',()=>{
  const cases=makeCases();assert.equal(cases.length,7*3*36);assert.equal(new Set(cases.map(c=>c.id)).size,cases.length);
  assert.equal(cases.filter(c=>c.variant.family==='native').length,63);
  assert.ok(cases.every(c=>c.iterations===4000&&c.resolution===1024));
  for(const material of materials)for(const seed of [1,2,3])assert.equal(cases.filter(c=>c.material===material&&c.seed===seed).length,36);
  assert.deepEqual(cases,makeCases());assert.equal(variants().filter(v=>v.id==='reference').length,1);
});
test('metrics exclude padding, normalize signed channels, and use correct all-mip weighting',()=>{
  const a=new Float32Array([0,0,0,0,0,0,0,0]);
  const b=new Float32Array([1,2,1,2,1,2,1,2]);
  const m=measure(a,b,2,[1,2]);assert.equal(m.normalizedMse,1);assert.equal(m.psnr,0);
  const exact=measure(a,a,2,[1,2]);assert.equal(exact.blockSsim,1);assert.equal(exact.psnr,null);
  const tiny=measure(new Float32Array([0,0]),new Float32Array([0.5,1]),1,[1,2]);
  const total=summarizeMips([m,tiny]);assert.equal(total.normalizedMse,4.25/5);assert.equal(total.equalMipMse,0.625);
  assert.throws(()=>measure(a,new Float32Array([NaN]),2,[1,2]));
});
test('uncertainty counts independent seeds and never fabricates a single-seed interval',()=>{
  assert.equal(pairedStats([1]).low,null);const result=pairedStats([1,2,3]);
  assert.equal(result.mean,2);assert.equal(result.sd,1);assert.equal(result.n,3);assert.ok(result.low<0&&result.high>4);
});

test('normal angular error follows positive-hemisphere reconstruction',()=>{
  assert.equal(normalError([0,0],[1,0],2,0).meanDegrees,90);
  assert.equal(normalError([0,0],[0,0],2,0).meanDegrees,0);
});

test('report marginal effects use complete contexts and paired seed replicates',async()=>{
  const {mkdtemp,mkdir,writeFile,readFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');const {join}=await import('node:path');
  const {generateReport}=await import('./report.mjs');
  const dir=await mkdtemp(join(tmpdir(),'ntc-report-test-'));
  try {
    await mkdir(join(dir,'results'));
    const cases=makeCases({onlyMaterials:['brick']}).filter(c=>c.variant.id==='reference'||c.variant.id.startsWith('pe-'));
    await writeFile(join(dir,'plan.json'),JSON.stringify({cases,seeds:[1,2,3],iterations:4000,resolution:1024,sourceHash:'synthetic-test',benchmark:'test'}));
    for(const c of cases) {
      const o={positionalEncoding:true,positionalEncodingPeriod:8,g0Sampling:'concatenate',quantization:{mode:'uint4'},...c.variant.changes};
      const tile=o.positionalEncoding&&o.positionalEncodingPeriod===8,cell=o.positionalEncoding&&!tile,concat=o.g0Sampling==='concatenate';
      const value=10+(tile?2:cell?1:0)+(concat?3:0)+(tile&&concat?1:0)+({uint2:0,uint4:4,uint8:5}[o.quantization.mode])+c.seed*0.1;
      const quality={psnr:value,mip0Psnr:value,equalMipPsnr:value,mips:[]};
      await writeFile(join(dir,'results',c.id+'.json'),JSON.stringify({...c,status:'complete',quality:{exported:quality},trainingMs:1,samples:1,storage:{},channels:[],adapter:{description:'test'},gradientPrecision:'float'}));
    }
    await generateReport(dir);
    const report=JSON.parse(await readFile(join(dir,'summary.json'),'utf8'));
    const tile=report.effects.find(e=>e.metric==='psnr'&&e.name==='encoding: none → tile');
    assert.equal(tile.n,3);assert.ok(Math.abs(tile.mean-2.5)<1e-10);assert.deepEqual(tile.contextsPerSeed,[6,6,6]);
    const interaction=report.effects.find(e=>e.metric==='psnr'&&e.kind==='interaction');
    assert.ok(Math.abs(interaction.mean-1)<1e-10);assert.equal(interaction.n,3);
  } finally {await rm(dir,{recursive:true,force:true});}
});
