"""Reproduce this report's derived data and figures. Requires matplotlib.
Run from any directory: python path/to/analyze.py [benchmark-directory]
Reads successful raw results without modifying the benchmark or its snapshot.
"""
import json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

out = Path(__file__).resolve().parent
root = out.parents[2]
source = Path(sys.argv[1]) if len(sys.argv)>1 else root/'output/benchmarks/main-profiles-4000'
raw = [json.loads(p.read_text()) for p in sorted((source/'results').glob('*.json')) if not p.name.endswith('.progress.json')]
records = [r for r in raw if r['status']=='complete']
materials = ['brick','uv_grid','road_aggregate','checkerboard_normal','standard_surface_rotate_scale2d_test','wood','velvet']
profiles = ['website','library','paper']
colors = dict(zip(profiles,['#007f83','#3464b4','#c35e26']))
names = {'standard_surface_rotate_scale2d_test':'rotate / scale 2D','checkerboard_normal':'checkerboard normal (constant source)'}
rows=[]
for r in records:
    assert r['completedIterations']==4000 and r['samples']==16384000
    peers=[x for x in records if x['material']==r['material']]
    assert all(x['referenceHashes']==r['referenceHashes'] for x in peers)
    assert all(x['adapter']==r['adapter'] for x in records)
    assert r['gradientPrecision']=='float' and not r['errors']
    assert not [e for e in r.get('sourceLog',[]) if e['severity']=='error']
    q=r['quality']['exported']; mip=q['mips'][0]
    properties=[]
    for c in r['channels']:
        cs=mip['perComponent'][c['offset']:c['offset']+c['size']]
        mse=sum(x['normalizedMse'] for x in cs)/len(cs)
        properties.append(dict(key=c['key'],mip0Psnr=-10*math.log10(mse) if mse else None,
            mip0ErrorShare=sum(x['normalizedMse'] for x in cs)/sum(x['normalizedMse'] for x in mip['perComponent']),
            constantAtMip0=all(x['referenceMin']==x['referenceMax'] for x in cs),
            normalMeanDegrees=mip.get('normals',{}).get(c['key'],{}).get('meanDegrees')))
    rows.append(dict(material=r['material'],profile=r['variant']['profile'],psnr=q['psnr'],normalizedMse=q['normalizedMse'],
        mip0Psnr=q['mip0Psnr'],equalMipPsnr=q['equalMipPsnr'],storage=r['storage'],trainingSeconds=r['trainingMs']/1000,
        exportDeltaDb=q['psnr']-r['quality']['trained']['psnr'],properties=properties,
        constantReference=all(c['referenceMin']==c['referenceMax'] for m in q['mips'] for c in m['perComponent']),
        decoderMacs=sum(a*b for a,b,_ in r['shape']['decoder']),
        payloadPlusWeightsBytes=r['storage']['latentPayloadBytes']+2*r['storage']['decoderParameters']))
comparisons=[]
for m in materials:
    rs={r['profile']:r for r in rows if r['material']==m}
    for a,b in [('website','library'),('library','paper')]:
        if b not in rs:continue
        x,y=rs[a],rs[b]
        comparisons.append(dict(material=m,baseline=a,candidate=b,deltaDb=y['psnr']-x['psnr'],
            errorReductionPercent=100*(1-y['normalizedMse']/x['normalizedMse']),fileRatio=y['storage']['jsonBytes']/x['storage']['jsonBytes']))
matched=[m for m in materials if m not in ['velvet','checkerboard_normal']]
aggregate={}
for p in profiles:
    rs=[r for r in rows if r['profile']==p and r['material'] in matched]
    mse=sum(r['normalizedMse'] for r in rs)/len(rs)
    aggregate[p]=dict(n=len(rs),normalizedMse=mse,psnrOfMeanMse=-10*math.log10(mse),meanTrainingSeconds=sum(r['trainingSeconds'] for r in rs)/len(rs))
data=dict(sourceHash=json.loads((source/'summary.json').read_text())['sourceHash'],records=rows,comparisons=comparisons,
    failed=[r['id'] for r in raw if r['status']!='complete'],aggregateMatchedNonconstant=aggregate)
(out/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(15,7.6),layout='constrained')
for ax,m in zip(axes.flat,materials):
    for r in [r for r in rows if r['material']==m]:
        p=r['profile'];x=r['storage']['jsonBytes']/1024;y=r['psnr']
        ax.scatter(x,y,color=colors[p],s=65,zorder=3,label=p)
        ax.annotate(f'{y:.2f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xscale('log');ax.set_xlim(24,800);ax.set_xticks([32,100,525],['32','100','525']);ax.xaxis.set_minor_locator(NullLocator())
    ys=[r['psnr'] for r in rows if r['material']==m];ax.set_ylim(min(ys)-2,max(ys)+3)
    ax.set_title(names.get(m,m),fontsize=10);ax.set_xlabel('Serialized JSON KiB (log scale)');ax.set_ylabel('Exported PSNR (dB)');ax.grid(alpha=.18)
axes.flat[-1].axis('off')
for p in profiles: axes.flat[-1].scatter([],[],color=colors[p],label=p,s=65)
axes.flat[-1].legend(loc='upper left',frameon=False)
axes.flat[-1].text(0,.57,'Higher and farther left is better.\nIndependent points, not fitted curves.\nPanel Y scales differ.\n\n4,000 iterations, batch 4,096, seed 1.\nVelvet / paper failed.\nCheckerboard source is constant.',transform=axes.flat[-1].transAxes,va='top',linespacing=1.6)
fig.suptitle('Main profile quality versus file size',fontsize=17)
fig.savefig(out/'rate-quality.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for ax,m in zip(axes.flat,[m for m in materials if m!='checkerboard_normal']):
    rs=[r for r in rows if r['material']==m]
    keys=[c['key'] for c in rs[0]['properties'] if not c['constantAtMip0']]
    for i,p in enumerate(profiles):
        r=next((r for r in rs if r['profile']==p),None)
        if not r:continue
        vals=[next(c['mip0Psnr'] for c in r['properties'] if c['key']==k) for k in keys]
        ax.bar([j+(i-1)*.24 for j in range(len(keys))],vals,width=.24,color=colors[p],label=p)
    ax.set_xticks(range(len(keys)),keys,rotation=20,ha='right');ax.set_ylim(0,60)
    ax.set_title(names.get(m,m));ax.set_ylabel('Mip-0 property PSNR (dB)');ax.grid(axis='y',alpha=.15)
axes.flat[0].legend(frameon=False)
fig.suptitle('Quality of spatially varying properties; constants excluded from this chart',fontsize=15)
fig.savefig(out/'property-quality.png',dpi=170);plt.close(fig)
print(f'Validated {len(rows)} successful records; wrote measurements.json and two figures to {out}')
