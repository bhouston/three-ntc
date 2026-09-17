"""Audit the completed study and reproduce its interpretation figures.
Run with Python + matplotlib from any directory. Does not change raw results.
"""
import gzip, hashlib, json, math, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

out=Path(__file__).resolve().parent
root=out.parents[2]
source=root/'output/benchmarks/feature-contributions-4000-v2'
plan=json.loads((source/'plan.json').read_text())
effect_data=json.loads((source/'feature-effects.json').read_text())
records=[]
for case in plan['cases']:
    r=json.loads((source/'results'/(case['id']+'.json')).read_text())
    assert r['status']=='complete' and r['completedIterations']==4000 and r['samples']==16384000
    assert r['sourceHash']==plan['sourceHash'] and not r['errors']
    assert not [e for e in r['sourceLog'] if e['severity']=='error']
    assert any(c['spatiallyVarying'] for c in r['sourceAudit'])
    model=(source/r['modelPath']).read_bytes()
    assert len(model)==r['storage']['jsonBytes']
    assert len((source/(r['modelPath']+'.gz')).read_bytes())==r['storage']['gzipBytes']
    assert len((source/(r['modelPath']+'.br')).read_bytes())==r['storage']['brotliBytes']
    assert gzip.decompress((source/(r['modelPath']+'.gz')).read_bytes())==model
    assert math.isfinite(r['quality']['exported']['psnr'])
    records.append(r)
assert len(records)==144
assert len({json.dumps(r['adapter'],sort_keys=True) for r in records})==1
assert {r['gradientPrecision'] for r in records}=={'float'}
materials=['brick','uv_grid','wood']
for m in materials:assert len({json.dumps(r['referenceHashes']) for r in records if r['material']==m})==1
labels={'reference':'Reference','pe-cell_g0-concatenate_uint4':'Cell-relative PE',
 'pe-none_g0-concatenate_uint4':'PE off','pe-tile_g0-bilinear_uint4':'G0 bilinear',
 'pe-none_g0-bilinear_uint4':'PE off + G0 bilinear','g1-off':'G1 off',
 'pe-tile_g0-concatenate_uint2':'2-bit latents','pe-tile_g0-concatenate_uint8':'8-bit latents',
 'hidden-16':'Hidden width 16','hidden-64':'Hidden width 64','activation-relu':'ReLU (width 32)',
 'relu-hidden-16':'ReLU + width 16','base-128':'Grid base 128','base-512':'Grid base 512',
 'grid-channels-8-12':'Grid channels 8/12','levels-2':'Two grid levels'}
aggregates=[]
for m in materials:
    for v in labels:
        rs=sorted([r for r in records if r['material']==m and r['variant']['id']==v],key=lambda r:r['seed'])
        assert [r['seed'] for r in rs]==[1,2,3]
        avg=lambda f:statistics.mean(f(r) for r in rs)
        aggregates.append(dict(material=m,variant=v,meanPsnr=avg(lambda r:r['quality']['exported']['psnr']),
            psnrOfMeanMse=-10*math.log10(avg(lambda r:r['quality']['exported']['normalizedMse'])),
            psnrBySeed=[r['quality']['exported']['psnr'] for r in rs],
            brotliKiB=avg(lambda r:r['storage']['brotliBytes']/1024),jsonKiB=avg(lambda r:r['storage']['jsonBytes']/1024),
            latentBytes=rs[0]['storage']['latentPayloadBytes'],runtimeGridBytes=rs[0]['storage']['runtimeGridBytes'],
            decoderMacs=sum(a*b for a,b,_ in rs[0]['shape']['decoder']),
            trainingSeconds=avg(lambda r:r['trainingMs']/1000),
            normalMeanDegrees=avg(lambda r:r['quality']['exported']['mips'][0]['normals']['normal']['meanDegrees']) if m!='uv_grid' else None))
# Independent paired-delta and t-interval checks against the generated findings.
for e in effect_data['effects']:
    base=next(a for a in aggregates if a['material']==e['material'] and a['variant']=='reference')
    variant=next(a for a in aggregates if a['material']==e['material'] and a['variant']==e['variant'])
    ds=[b-a for a,b in zip(base['psnrBySeed'],variant['psnrBySeed'])]
    assert abs(statistics.mean(ds)-e['mean'])<1e-10
    assert abs(e['low']-(statistics.mean(ds)-4.302652730*statistics.stdev(ds)/math.sqrt(3)))<1e-9
summary=dict(sourceHash=plan['sourceHash'],verifiedRuns=len(records),nominalSamples=sum(r['samples'] for r in records),
 trainingHours=sum(r['trainingMs'] for r in records)/3600000,aggregates=aggregates,
 analysisSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'analysis-data.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
order=[v for v in labels if v!='reference']
values=[[next(e['mean'] for e in effect_data['effects'] if e['variant']==v and e['material']==m) for m in materials] for v in order]
fig,ax=plt.subplots(figsize=(9,9),layout='constrained')
im=ax.imshow(values,cmap='BrBG',vmin=-6.5,vmax=6.5,aspect='auto')
ax.set_xticks(range(3),['Brick','UV grid','Wood']);ax.set_yticks(range(len(order)),[labels[v] for v in order])
for i,v in enumerate(order):
    for j,m in enumerate(materials):
        e=next(e for e in effect_data['effects'] if e['variant']==v and e['material']==m)
        ax.text(j,i,f"{e['mean']:+.2f}"+('*' if e.get('holmP',1)<.05 else ''),ha='center',va='center',color='white' if abs(e['mean'])>3.5 else '#152522')
ax.set_title('Feature changes relative to the controlled reference\nMean paired PSNR difference (dB), three seeds',pad=14)
fig.colorbar(im,ax=ax,label='Higher is better')
fig.supxlabel('* Holm p < 0.05 within material. No star does not establish equivalence.',fontsize=10)
fig.savefig(out/'contribution-summary.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,5.7),layout='constrained')
selected={'pe-cell_g0-concatenate_uint4':('Cell PE','#07827e'), 'hidden-64':('Width 64','#8552a4'),
 'base-512':('Base 512','#b66027'),'base-128':('Base 128','#b66027'),
 'pe-tile_g0-concatenate_uint8':('8 bits','#3464b4'),'pe-tile_g0-concatenate_uint2':('2 bits','#3464b4'),
 'reference':('Ref.','#222222')}
for ax,m in zip(axes,materials):
    rs=[r for r in aggregates if r['material']==m]
    for r in rs:
        v=r['variant'];x=r['brotliKiB'];y=r['psnrOfMeanMse']
        ax.scatter(x,y,color=selected.get(v,('', '#b6bbc0'))[1],s=48 if v in selected else 25,zorder=3)
        if v in selected:
            offset=(0,9)
            if v=='reference':offset=(0,-18)
            if v=='hidden-64' and m=='wood':offset=(0,-19)
            ax.annotate(selected[v][0],(x,y),xytext=offset,textcoords='offset points',ha='center',fontsize=9)
    ax.set_xscale('log');ax.set_xlim(36,850);ax.set_xticks([50,100,200,400,800],['50','100','200','400','800']);ax.xaxis.set_minor_locator(NullLocator())
    ys=[r['psnrOfMeanMse'] for r in rs];ax.set_ylim(min(ys)-1.1,max(ys)+1.2)
    ax.set_title(m);ax.set_xlabel('Actual Brotli KiB (log scale)');ax.set_ylabel('PSNR of mean seed MSE (dB)');ax.grid(alpha=.17)
fig.suptitle('Measured quality versus download size; each point is a trained configuration',fontsize=14)
fig.supxlabel('Higher and farther left is better. Gray points are other tested configurations; panel Y scales differ.',fontsize=10)
fig.savefig(out/'rate-quality.png',dpi=170);plt.close(fig)
print(f'Audited {len(records)} results, stored byte counts, gzip roundtrips and {len(effect_data["effects"])} paired contrasts. Training: {summary["trainingHours"]:.2f} hours.')
