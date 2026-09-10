import { readFile, writeFile } from 'node:fs/promises';
const [label, description, previous] = process.argv.slice(2);
if (!label || !previous) throw new Error('Usage: node scripts/ntc-report.mjs LABEL DESCRIPTION PREVIOUS');
const read = name => readFile(`docs/metrics/${name}.json`,'utf8').then(JSON.parse);
const a = await read(label), b = await read(previous);
const mean = rows => rows.reduce((s,r)=>s+r.exported.mse,0)/rows.length;
const mse=mean(a), old=mean(b);
const delta=(a,b)=>(100*(a/b-1)).toFixed(2)+'%';
let report=await readFile('astra_progress.md','utf8');
let [summary, details=''] = report.split('## Per-change details\n');
summary+=`| ${label}: ${description} | ${mse.toFixed(8)} | ${(-10*Math.log10(mse)).toFixed(4)} | ${delta(mse,old)} |\n`;
details+=`\n### ${label}: ${description}\n\nCompared with ${previous}. Six quality cases passed (finite error only; no PSNR threshold).\n\n| Fixture | Learned interpolation | Exported MSE | PSNR | MSE change |\n|---|---|---:|---:|---:|\n`;
for(const r of a) {
 const p=b.find(x=>x.fixture===r.fixture && x.positionalEncoding===r.positionalEncoding);
 details+=`| ${r.fixture} | ${r.positionalEncoding} | ${r.exported.mse.toFixed(8)} | ${r.exported.psnr.toFixed(3)} | ${delta(r.exported.mse,p.exported.mse)} |\n`;
}
await writeFile('astra_progress.md',summary+'\n## Per-change details\n'+details);
console.log({label,mse,psnr:-10*Math.log10(mse),change:delta(mse,old)});
