import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { pairedStats } from './metrics.mjs';

export async function generateReport(out) {
  const plan=JSON.parse(await readFile(join(out,'plan.json'),'utf8'));
  const records=[];
  for(const file of await readdir(join(out,'results')))if(file.endsWith('.json')&&!file.endsWith('.progress.json'))records.push(JSON.parse(await readFile(join(out,'results',file),'utf8')));
  const good=records.filter(r=>r.status==='complete'),failed=records.filter(r=>r.status==='failed');
  const byId=new Map(good.map(r=>[r.id,r]));
  const profileSummaries=[];
  for(const material of new Set(plan.cases.map(c=>c.material)))for(const id of new Set(plan.cases.filter(c=>c.variant.profile).map(c=>c.variant.id)))for(const metric of ['psnr','mip0Psnr','equalMipPsnr']) {
    const rows=good.filter(r=>r.material===material&&r.variant.id===id&&typeof r.quality.exported[metric]==='number');
    const stats=pairedStats(rows.map(r=>r.quality.exported[metric]));
    if(stats)profileSummaries.push({material,name:id,metric,kind:'profiles',...stats,completeSeeds:stats.n===plan.seeds.length});
  }
  const effects=[];
  const metricNames=['psnr','mip0Psnr','equalMipPsnr'];
  const ids=[...new Set(plan.cases.map(c=>c.variant.id))];
  const materials=[...new Set(plan.cases.map(c=>c.material))];
  const row=(material,name,metric,pairs,kind)=>{
    // Average crossed contexts within each seed before estimating seed uncertainty.
    const perSeed=new Map();
    for(const [a,b] of pairs) {
      if(JSON.stringify(a.adapter)!==JSON.stringify(b.adapter)||a.gradientPrecision!==b.gradientPrecision||JSON.stringify(a.referenceHashes)!==JSON.stringify(b.referenceHashes))continue;
      if(typeof a.quality.exported[metric]!=='number'||typeof b.quality.exported[metric]!=='number')continue;
      const value=b.quality.exported[metric]-a.quality.exported[metric];
      if(!Number.isFinite(value))continue;
      const group=perSeed.get(a.seed)??[];group.push(value);perSeed.set(a.seed,group);
    }
    const values=[...perSeed.values()].map(v=>v.reduce((a,b)=>a+b,0)/v.length);
    const stats=pairedStats(values);if(stats)effects.push({material,name,metric,kind,...stats,
      completeSeeds:stats.n===plan.seeds.length,contextsPerSeed:[...perSeed.values()].map(v=>v.length)});
  };
  for(const material of materials)for(const metric of metricNames) {
    for(const id of ids.filter(id=>id!=='reference'&&!id.endsWith('-native'))) {
      const pairs=plan.seeds.map(seed=>[byId.get(`${material}__reference__seed-${seed}`),byId.get(`${material}__${id}__seed-${seed}`)]).filter(pair=>pair.every(Boolean));
      row(material,id,metric,pairs,'vs-reference');
    }
    const factorialId=(pe,g0,bits)=>pe==='tile'&&g0==='concatenate'&&bits===4?'reference':`pe-${pe}_g0-${g0}_uint${bits}`;
    const match=(a,b)=>plan.seeds.flatMap(seed=>{
      const x=byId.get(`${material}__${a}__seed-${seed}`),y=byId.get(`${material}__${b}__seed-${seed}`);return x&&y?[[x,y]]:[];
    });
    const interactions=[];
    for(const seed of plan.seeds)for(const bits of [2,4,8]) {
      const get=(pe,g0)=>byId.get(`${material}__${factorialId(pe,g0,bits)}__seed-${seed}`);
      const a=get('none','bilinear'),b=get('none','concatenate'),c=get('tile','bilinear'),d=get('tile','concatenate');
      if([a,b,c,d].every(r=>r&&typeof r.quality.exported[metric]==='number'&&JSON.stringify(r.adapter)===JSON.stringify(a.adapter)&&r.gradientPrecision===a.gradientPrecision&&JSON.stringify(r.referenceHashes)===JSON.stringify(a.referenceHashes))) {
        const difference=(x,y)=>({...x,quality:{exported:{[metric]:y.quality.exported[metric]-x.quality.exported[metric]}}});
        interactions.push([difference(a,b),difference(c,d)]);
      }
    }
    const interactionCounts=new Map();for(const [a] of interactions)interactionCounts.set(a.seed,(interactionCounts.get(a.seed)??0)+1);
    row(material,'G0 benefit: tiled PE minus no PE',metric,interactions.filter(([a])=>interactionCounts.get(a.seed)===3),'interaction');
    const widthPairs=[];
    for(const seed of plan.seeds) {
      const get=id=>byId.get(`${material}__${id}__seed-${seed}`);
      const a=get('reference'),b=get('hidden-16'),c=get('activation-relu'),d=get('relu-hidden-16');
      if([a,b,c,d].every(r=>r&&typeof r.quality.exported[metric]==='number'&&JSON.stringify(r.adapter)===JSON.stringify(a.adapter)&&r.gradientPrecision===a.gradientPrecision&&JSON.stringify(r.referenceHashes)===JSON.stringify(a.referenceHashes))) {
        const difference=(x,y)=>({...x,quality:{exported:{[metric]:y.quality.exported[metric]-x.quality.exported[metric]}}});
        widthPairs.push([difference(c,d),difference(a,b)]);
      }
    }
    row(material,'Width 16 effect: hardGELU minus ReLU',metric,widthPairs,'interaction');
    const factorialEffects=[
      ['encoding: none → tile', ['bilinear','concatenate'].flatMap(g0=>[2,4,8].flatMap(bits=>match(factorialId('none',g0,bits),factorialId('tile',g0,bits)))),6],
      ['encoding: cell → tile', ['bilinear','concatenate'].flatMap(g0=>[2,4,8].flatMap(bits=>match(factorialId('cell',g0,bits),factorialId('tile',g0,bits)))),6],
      ['G0: bilinear → concatenate', ['none','cell','tile'].flatMap(pe=>[2,4,8].flatMap(bits=>match(factorialId(pe,'bilinear',bits),factorialId(pe,'concatenate',bits)))),9],
      ['quantization: uint2 → uint4', ['none','cell','tile'].flatMap(pe=>['bilinear','concatenate'].flatMap(g0=>match(factorialId(pe,g0,2),factorialId(pe,g0,4)))),6],
      ['quantization: uint4 → uint8', ['none','cell','tile'].flatMap(pe=>['bilinear','concatenate'].flatMap(g0=>match(factorialId(pe,g0,4),factorialId(pe,g0,8)))),6],
    ];
    for(const [name,pairs,contexts] of factorialEffects) {
      // Marginal effects are only emitted for seeds with the whole crossing.
      const counts=new Map();for(const [a] of pairs)counts.set(a.seed,(counts.get(a.seed)??0)+1);
      row(material,name,metric,pairs.filter(([a])=>counts.get(a.seed)===contexts),'factorial');
    }
  }
  const profileOnly=plan.cases.every(c=>c.variant.profile);
  const summary={profileOnly,sourceHash:plan.sourceHash,benchmark:plan.benchmark,iterations:plan.iterations,resolution:plan.resolution,
    expected:plan.cases.length,complete:good.length,failed:failed.length,pending:plan.cases.length-records.length,
    final:good.length===plan.cases.length,materials,seeds:plan.seeds,effects,profileSummaries,
    records:good.map(r=>({id:r.id,material:r.material,variant:r.variant,seed:r.seed,quality:r.quality.exported,
      channels:r.channels,trainingMs:r.trainingMs,samples:r.samples,storage:r.storage,preview:r.preview,shape:r.shape,adapter:r.adapter,gradientPrecision:r.gradientPrecision}))};
  await writeFile(join(out,'summary.json'),JSON.stringify(summary,null,2));
  const csv=(rows)=>rows.map(r=>r.map(v=>JSON.stringify(v??'')).join(',')).join('\n')+'\n';
  await writeFile(join(out,'contributions.csv'),csv([['material','comparison','metric','kind','seeds','meanDeltaDb','ci95Low','ci95High','completeSeeds'],...effects.map(e=>[e.material,e.name,e.metric,e.kind,e.n,e.mean,e.low,e.high,e.completeSeeds])]));
  await writeFile(join(out,'runs.csv'),csv([['material','variant','seed','psnr','mip0Psnr','equalMipPsnr','trainingSeconds','samples','jsonBytes','latentBytes'],...good.map(r=>[r.material,r.variant.id,r.seed,r.quality.exported.psnr,r.quality.exported.mip0Psnr,r.quality.exported.equalMipPsnr,r.trainingMs/1000,r.samples,r.storage.jsonBytes,r.storage.latentPayloadBytes])]));
  const components=[['material','variant','seed','mip','property','component','mse','normalizedMse','mae','maxError','blockSsim']];
  for(const r of good)for(const [lod,m] of r.quality.exported.mips.entries())for(const ch of r.channels)for(let k=0;k<ch.size;k++) {
    const q=m.perComponent[ch.offset+k];components.push([r.material,r.variant.id,r.seed,lod,ch.key,k,q.mse,q.normalizedMse,q.mae,q.maxError,q.blockSsim]);
  }
  await writeFile(join(out,'per-property-mip.csv'),csv(components));
  const normals=[['material','variant','seed','mip','property','meanDegrees','rmsDegrees','maxDegrees','outsideUnitDiskFraction']];
  for(const r of good)for(const [lod,m] of r.quality.exported.mips.entries())for(const [key,q] of Object.entries(m.normals??{}))
    normals.push([r.material,r.variant.id,r.seed,lod,key,q.meanDegrees,q.rmsDegrees,q.maxDegrees,q.outsideUnitDiskFraction]);
  await writeFile(join(out,'normal-errors.csv'),csv(normals));
  const lines=[`MaterialX ${profileOnly?'profile':'contribution'} benchmark (${summary.final?'complete':'IN PROGRESS'})`, '',
    `${good.length}/${plan.cases.length} complete; ${failed.length} failed; ${summary.pending} pending.`,
    `${plan.iterations} iterations/run; ${plan.resolution} × ${plan.resolution}; seeds ${plan.seeds.join(', ')}.`,
    `Source snapshot: ${plan.sourceHash}. Mode: ${plan.benchmark}.`, '',
    'Positive paired ΔPSNR means the named variation improves quality. Confidence intervals use seed replicates, not texels. Three seeds give broad intervals; exploratory comparisons are not adjusted for multiple testing. Effects are conditional and are not additive percentages of quality.', '',
    'The primary metric is exported-model PSNR of range-normalized errors over every active component and every texel in the complete mip chain. Mip-0 PSNR and equal-mip PSNR are also reported. Fixed range is 2 for signed offsets and 1 otherwise; HDR uses a unit reference. Block SSIM uses disjoint 8×8 population-statistic windows. Reference mips use the common source box chain; the native paper profile trains with Lanczos.', '',
    'Controlled feature runs use STE on both grids. Profile behavior is determined by the recorded source snapshot; do not mix snapshots when comparing effects. Equal-budget profiles change only batch size and iteration count. Native runs retain batch sizes, with exact logical batches chunked for memory and one optimizer update per batch.', '',
    'Charts: report.html. Raw records: results/. Per-property/per-mip metrics: per-property-mip.csv. Preview panels: source | decoded | absolute error ×10 (linear display, one representative channel).', '',];
  for(const material of materials) {
    lines.push(`**${material}.mtlx**`, '');
    const rows=effects.filter(e=>e.material===material&&e.metric==='psnr'&&e.kind==='factorial');
    if(!rows.length)lines.push('Factorial comparisons pending complete matched contexts.','');
    for(const e of rows.sort((a,b)=>b.mean-a.mean))lines.push(`- ${e.name}: ${e.mean.toFixed(3)} dB (${e.n} seeds${e.low===null?'':`, 95% CI ${e.low.toFixed(3)} to ${e.high.toFixed(3)}`}).`);
    lines.push('', 'Conditional changes from the controlled reference:', '',
      '| Variation | Mean ΔPSNR (dB) | 95% paired interval | Seeds |',
      '| --- | ---: | --- | ---: |');
    for(const e of effects.filter(e=>e.material===material&&e.metric==='psnr'&&e.kind==='vs-reference').sort((a,b)=>b.mean-a.mean))
      lines.push(`| ${e.name} | ${e.mean.toFixed(3)} | ${e.low===null?'pending':`${e.low.toFixed(3)} to ${e.high.toFixed(3)}`} | ${e.n}/${plan.seeds.length} |`);
    lines.push('', 'Profile bundle measurements (unequal native sample budgets are not isolated feature effects):', '');
    for(const id of ids.filter(id=>id.startsWith('profile-'))) {
      const runs=good.filter(r=>r.material===material&&r.variant.id===id);
      if(!runs.length)continue;
      const q=pairedStats(runs.map(r=>r.quality.exported.psnr).filter(v=>typeof v==='number'));
      if(q)lines.push(`- ${id}: mean ${q.mean.toFixed(3)} dB, ${q.n}/${plan.seeds.length} seeds, ${runs[0].samples.toLocaleString()} samples/run, mean training ${(runs.reduce((s,r)=>s+r.trainingMs,0)/runs.length/1000).toFixed(1)} s.`);
    }
    lines.push('');
  }
  if(failed.length)lines.push('Failed runs: '+failed.map(r=>r.id).join(', '));
  await writeFile(join(out,'analysis.md'),lines.join('\n'));
  const data=JSON.stringify(summary).replaceAll('<','\\u003c');
  await writeFile(join(out,'report.html'),`<!doctype html><html><head><meta charset="utf-8"><title>NTC feature contribution benchmark</title><style>
body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#fafafa;color:#17202a}h1{font-size:28px}select{padding:8px;margin:10px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd;font-variant-numeric:tabular-nums}svg{background:white;width:100%}.note{color:#596473;max-width:1100px;line-height:1.6}.status{padding:16px;background:#e9f0ff}img{width:768px;max-width:100%}a{color:#1760a5}</style></head><body>
<h1>${profileOnly?'Main profiles: material quality':'Which features improve each material?'}</h1><p id="status" class="status"></p>
<p class="note">${profileOnly?'Main profiles at the same 4,000-iteration / 4,096-sample budget. Higher PSNR is better. Single-seed results are preliminary; there is no confidence interval yet. This compares each complete profile, not individual features.':'4,000-iteration production results are kept separate from smoke checks. Positive effect bars mean better exported-material quality. Intervals use seed replicates, not pixels. Effects are not additive shares of quality.'} Reload to see new completed runs.</p>
<label>Material <select id="material"></select></label><label>Metric <select id="metric"><option value="psnr">All-texel PSNR</option><option value="mip0Psnr">Mip-0 PSNR</option><option value="equalMipPsnr">Equal-mip PSNR</option></select></label>
<label>Comparison <select id="kind"><option value="profiles">Main profiles: absolute quality</option><option value="factorial">Factorial marginal effects</option><option value="vs-reference">Variants versus controlled reference</option><option value="interaction">Feature interactions (difference of effects)</option></select></label>
<p class="note">${profileOnly?'Library defaults, website defaults, and getNTCPaperProfile, with the same iteration and batch budgets. Metrics use exported/reloaded models, all active properties, and a common source mip chain.':'Factorial effects average matched encoding × sampling × quantization contexts within each seed. Other rows compare against the controlled reference. Profile comparisons change multiple settings and are not individual feature contributions.'}</p>
<div id="chart"></div><div id="effectTable"></div><h2>Completed runs</h2><div id="runs"></div><h2>Property error metrics</h2><label>Mip <select id="propertyMip"></select></label><div id="properties"></div><h2>Visual comparison</h2><select id="preview"></select><p>Source | decoded | absolute error ×10; a representative material channel.</p><img id="image"><p id="channel"></p>
<p><a href="analysis.md">Analysis</a> · <a href="runs.csv">Run metrics CSV</a> · <a href="contributions.csv">Contribution CSV</a> · <a href="per-property-mip.csv">Every property and mip</a> · <a href="plan.json">Reproducibility manifest</a></p>
<script>const data=${data};const $=id=>document.getElementById(id);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('status').textContent=data.complete+'/'+data.expected+' complete · '+data.failed+' failed · '+data.pending+' pending · '+data.benchmark+' · '+data.iterations+' iterations · '+data.resolution+'px · source '+data.sourceHash.slice(0,12)+(data.final?' · COMPLETE':' · PRELIMINARY');
$('material').innerHTML=data.materials.map(m=>'<option>'+esc(m)+'</option>').join('');const fmt=v=>v==null?'—':Number(v).toFixed(3);
function draw(){const material=$('material').value,metric=$('metric').value,kind=$('kind').value;const rows=(kind==='profiles'?data.profileSummaries:data.effects).filter(e=>e.material===material&&e.metric===metric&&e.kind===kind).sort((a,b)=>b.mean-a.mean);const extent=Math.max(1,...rows.flatMap(e=>[Math.abs(e.mean),Math.abs(e.low??0),Math.abs(e.high??0)]));const x=v=>800+v/extent*350;let svg='<svg viewBox="0 0 1200 '+Math.max(100,rows.length*36+55)+'"><line x1="800" y1="20" x2="800" y2="'+(rows.length*36+30)+'" stroke="#999"/>';
rows.forEach((e,i)=>{const y=25+i*36;svg+='<text x="8" y="'+(y+16)+'" font-size="14">'+esc(e.name)+'</text><rect x="'+Math.min(800,x(e.mean))+'" y="'+y+'" width="'+Math.max(1,Math.abs(x(e.mean)-800))+'" height="23" fill="'+(e.mean>=0?'#258c79':'#c1605a')+'"/>';if(e.low!==null)svg+='<line x1="'+x(e.low)+'" x2="'+x(e.high)+'" y1="'+(y+12)+'" y2="'+(y+12)+'" stroke="#192b44" stroke-width="2"/>';});svg+='<text x="800" y="'+(rows.length*36+48)+'" font-size="13">'+(kind==='profiles'?'0 · PSNR (dB)':'0 · ΔPSNR (dB)')+'</text></svg>';$('chart').innerHTML=rows.length?svg:'Waiting for matched runs.';
$('effectTable').innerHTML='<table><tr><th>Comparison</th><th>'+ (kind==='profiles'?'Mean PSNR dB':'Mean ΔdB') +'</th><th>95% interval</th><th>Seeds</th></tr>'+rows.map(e=>'<tr><td>'+esc(e.name)+'</td><td>'+fmt(e.mean)+'</td><td>'+fmt(e.low)+' to '+fmt(e.high)+'</td><td>'+e.n+(e.completeSeeds?'':' (partial)')+'</td></tr>').join('')+'</table>';
const runs=data.records.filter(r=>r.material===material).sort((a,b)=>a.variant.id.localeCompare(b.variant.id)||a.seed-b.seed);
$('runs').innerHTML='<table><tr><th>Variant</th><th>Seed</th><th>PSNR dB</th><th>Training s</th><th>Samples</th><th>File KiB</th></tr>'+runs.map(r=>'<tr><td>'+esc(r.variant.id)+'</td><td>'+r.seed+'</td><td>'+fmt(r.quality[metric])+'</td><td>'+fmt(r.trainingMs/1000)+'</td><td>'+r.samples.toLocaleString()+'</td><td>'+fmt(r.storage.jsonBytes/1024)+'</td></tr>').join('')+'</table>';
const previousMip=$('propertyMip').value;const mipCount=runs[0]?.quality.mips.length??0;
$('propertyMip').innerHTML=Array.from({length:mipCount},(_,i)=>'<option>'+i+'</option>').join('');if(Number(previousMip)<mipCount)$('propertyMip').value=previousMip||'0';drawProperties();
$('preview').innerHTML=runs.filter(r=>r.preview).map(r=>'<option value="'+esc(r.id)+'">'+esc(r.variant.id)+' / seed '+r.seed+'</option>').join('');showPreview();}
function drawProperties(){const material=$('material').value,lod=Number($('propertyMip').value||0);const rows=[];
for(const r of data.records.filter(r=>r.material===material))for(const ch of r.channels??[]) {
 const m=r.quality.mips[lod];if(!m)continue;const cs=m.perComponent.slice(ch.offset,ch.offset+ch.size);
 const mse=cs.reduce((s,c)=>s+c.mse,0)/cs.length,nmse=cs.reduce((s,c)=>s+c.normalizedMse,0)/cs.length;
 rows.push('<tr><td>'+esc(r.variant.id)+' / '+r.seed+'</td><td>'+esc(ch.key)+'</td><td>'+mse.toExponential(4)+'</td><td>'+fmt(nmse>0?-10*Math.log10(nmse):null)+'</td><td>'+fmt(cs.reduce((s,c)=>s+c.blockSsim,0)/cs.length)+'</td><td>'+fmt(m.normals?.[ch.key]?.meanDegrees)+'</td></tr>');
}
$('properties').innerHTML='<table><tr><th>Profile / seed</th><th>Property</th><th>MSE</th><th>Normalized PSNR</th><th>Block SSIM</th><th>Normal mean °</th></tr>'+rows.join('')+'</table>';}
function showPreview(){const r=data.records.find(r=>r.id===$('preview').value);$('image').src=r?.preview?.path??'';$('channel').textContent=r?'Channel: '+r.preview.channel:'';}
for(const id of ['material','metric','kind'])$(id).onchange=draw;$('preview').onchange=showPreview;$('propertyMip').onchange=drawProperties;draw();</script></body></html>`);
}
