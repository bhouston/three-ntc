"""Update a controlled study's Markdown findings and plots from accepted results.
Requires matplotlib. Never runs training or modifies raw results.
"""
import json, math, sys, statistics, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]).resolve()
plan=json.loads((out/'plan.json').read_text())
raw=[json.loads(p.read_text()) for p in sorted((out/'results').glob('*.json')) if not p.name.endswith('.progress.json')]
good=[r for r in raw if r.get('status')=='complete']
failed=[r for r in raw if r.get('status')=='failed']
by={(r['material'],r['variant']['id'],r['seed']):r for r in good}
materials=list(dict.fromkeys(c['material'] for c in plan['cases']))
variants=list(dict.fromkeys(c['variant']['id'] for c in plan['cases']))
seeds=plan['seeds']
labels={'reference':'Reference','pe-none_g0-concatenate_uint4':'PE off','pe-cell_g0-concatenate_uint4':'Cell PE',
 'pe-tile_g0-bilinear_uint4':'G0 bilinear','pe-none_g0-bilinear_uint4':'PE off + G0 bilinear',
 'pe-tile_g0-concatenate_uint2':'2-bit latents','pe-tile_g0-concatenate_uint8':'8-bit latents',
 'activation-relu':'ReLU','hidden-16':'Width 16','hidden-64':'Width 64','relu-hidden-16':'ReLU + width 16',
 'g1-off':'G1 off','base-128':'G0 base 128','base-512':'G0 base 512','levels-2':'Two grid levels',
 'grid-channels-8-12':'Channels 8 / 12'}

def compatible(rs):
    return all(r['sourceHash']==plan['sourceHash'] and r['completedIterations']==plan['iterations']
       and r['samples']==plan['iterations']*4096
       and r['referenceHashes']==rs[0]['referenceHashes'] and r['adapter']==rs[0]['adapter']
       and r['gradientPrecision']==rs[0]['gradientPrecision'] for r in rs)
def stats(values):
    if not values:return None
    n=len(values);mean=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(n) if n>1 else None
    critical={2:12.706204736,3:4.302652730}.get(n)
    half=critical*se if critical is not None else None
    p=None
    if se is not None:
        if se==0:p=0.0 if mean else 1.0
        elif n==3:p=1-abs(mean/se)/math.sqrt((mean/se)**2+2)
        elif n==2:p=1-2/math.pi*math.atan(abs(mean/se))
    return dict(n=n,mean=mean,low=mean-half if half is not None else None,high=mean+half if half is not None else None,p=p,values=values)
def prop_psnr(r,ch):
    mips=r['quality']['exported']['mips'];total=sum(m['pixels'] for m in mips)
    mse=sum(m['pixels']*sum(c['normalizedMse'] for c in m['perComponent'][ch['offset']:ch['offset']+ch['size']])/ch['size'] for m in mips)/total
    return -10*math.log10(mse) if mse>0 else None

effects=[];property_effects=[];interactions=[]
for material in materials:
    for variant in variants:
        if variant=='reference':continue
        pairs=[]
        for seed in seeds:
            a=by.get((material,'reference',seed));b=by.get((material,variant,seed))
            if a and b and compatible([a,b]):pairs.append((a,b))
        if not pairs:continue
        e=stats([b['quality']['exported']['psnr']-a['quality']['exported']['psnr'] for a,b in pairs])
        e.update(material=material,variant=variant,complete=e['n']==len(seeds),
            geometricMseReductionPercent=100*(1-10**(-e['mean']/10)),
            jsonRatio=statistics.mean(b['storage']['jsonBytes']/a['storage']['jsonBytes'] for a,b in pairs),
            brotliRatio=statistics.mean(b['storage']['brotliBytes']/a['storage']['brotliBytes'] for a,b in pairs) if all('brotliBytes' in r['storage'] for pair in pairs for r in pair) else None,
            runtimeGridRatio=statistics.mean(b['storage']['runtimeGridBytes']/a['storage']['runtimeGridBytes'] for a,b in pairs),
            trainingRatio=statistics.mean(b['trainingMs']/a['trainingMs'] for a,b in pairs),
            mip0Delta=statistics.mean(b['quality']['exported']['mip0Psnr']-a['quality']['exported']['mip0Psnr'] for a,b in pairs),
            equalMipDelta=statistics.mean(b['quality']['exported']['equalMipPsnr']-a['quality']['exported']['equalMipPsnr'] for a,b in pairs))
        effects.append(e)
        for ch in pairs[0][0]['channels']:
            values=[prop_psnr(b,ch)-prop_psnr(a,ch) for a,b in pairs if prop_psnr(a,ch) is not None and prop_psnr(b,ch) is not None]
            st=stats(values)
            if st:property_effects.append(dict(material=material,variant=variant,property=ch['key'],
                spatiallyVarying=next((c['spatiallyVarying'] for c in pairs[0][0].get('sourceAudit',[]) if c['key']==ch['key']),True),**st))
    # Two predefined difference-of-differences. Do not pool missing bit contexts.
    for name,ids in [
      ('G0 concatenation benefit: tiled PE minus PE off', ['pe-none_g0-bilinear_uint4','pe-none_g0-concatenate_uint4','pe-tile_g0-bilinear_uint4','reference']),
      ('Width-16 effect: hardGELU minus ReLU',['activation-relu','relu-hidden-16','reference','hidden-16'])]:
        vals=[]
        for seed in seeds:
            rs=[by.get((material,v,seed)) for v in ids]
            if all(rs) and compatible(rs):
                a,b,c,d=[r['quality']['exported']['psnr'] for r in rs];vals.append((d-c)-(b-a))
        st=stats(vals)
        if st:interactions.append(dict(material=material,name=name,**st))
# Holm correction across all planned reference contrasts per material. Missing
# comparisons count toward the family size, preventing partial-run discoveries.
for material in materials:
    family=[e for e in effects if e['material']==material and e['complete'] and e['p'] is not None]
    maximum=0
    for i,e in enumerate(sorted(family,key=lambda e:e['p'])):
        maximum=max(maximum,min(1,e['p']*(len(variants)-1-i)));e['holmP']=maximum

finished=len(raw)==len(plan['cases'])
lines=['# Controlled feature study: findings', '',
 f"Updated from source `{plan['sourceHash']}`.", '',
 f"**{'Queue finished' if finished else 'Interim results'}:** {len(good)}/{len(plan['cases'])} successful; {len(failed)} failed; {len(plan['cases'])-len(raw)} pending. {plan['iterations']} iterations, batch 4096, resolution {plan['resolution']}, seeds {seeds}.", '',
 'Every delta changes the named configuration relative to a shared reference: paired 4/4-channel grids, base 256, four levels, G0 concatenation, eight-texel PE, width 32 hardGELU, 4-bit STE. Source and optimizer settings are fixed. Width changes refer to both hidden layers.', '',
 '**Interpretation:** positive ΔPSNR means the change helps; negative means the removed/replaced reference feature helped in this context. Capacity changes also change storage. These are conditional effects, not additive percentages or a ranking valid at every bitrate. Pairing checks source-mip hashes, snapshot, sample count, GPU and gradient precision.', '',
 'Intervals are unadjusted paired Student-t 95% intervals across seeds. With three seeds they can be broad. Holm p-values adjust the primary reference contrasts within each material; interaction and property statistics remain exploratory. A complete-seed result with Holm p<0.05 is evidence within this study, not proof of generality. Missing cases are never imputed. Timing ratios include compilation and are not decoder performance.', '',
 'A geometric MSE reduction is derived from the mean paired dB difference. Brotli sizes use quality 5 on the serialized model. Grid residency is RGBA16F texel accounting, not total process memory.', '']
for material in materials:
    rows=sorted([e for e in effects if e['material']==material],key=lambda e:e['mean'],reverse=True)
    lines += [f'**{material}.mtlx**','', '| Change | ΔPSNR dB | 95% interval | Seeds | MSE reduction | Brotli ratio | Grid memory ratio | Holm p |',
      '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for e in rows:
        ci='pending' if e['low'] is None else f"{e['low']:+.2f} to {e['high']:+.2f}"
        br='unavailable' if e['brotliRatio'] is None else f"{e['brotliRatio']:.2f}×"
        hp=f"{e['holmP']:.4f}" if 'holmP' in e else 'pending'
        lines.append(f"| {labels.get(e['variant'],e['variant'])} | {e['mean']:+.3f} | {ci} | {e['n']}/{len(seeds)} | {e['geometricMseReductionPercent']:+.1f}% | {br} | {e['runtimeGridRatio']:.2f}× | {hp} |")
    if not rows:lines+=['','Waiting for a completed reference and a matching variant.']
    supported=[e for e in rows if e.get('holmP',1)<.05]
    lines+=['']
    if supported:
        lines+=['Evidence passing the within-material correction: '+ '; '.join(f"{labels[e['variant']]} {'improves' if e['mean']>0 else 'reduces'} quality by {abs(e['mean']):.2f} dB" for e in supported)+'.']
    else:lines+=['No completed effect yet passes the predefined within-material correction. Inspect the estimates and intervals without interpreting this as proof of equivalence.']
    lines+=['','Paired mip-0 / equal-mip effects (dB): '+('; '.join(f"{labels[e['variant']]} {e['mip0Delta']:+.2f} / {e['equalMipDelta']:+.2f}" for e in rows) or 'pending')+'.','']
    for e in [e for e in interactions if e['material']==material]:
        lines.append(f"Interaction: {e['name']}: {e['mean']:+.3f} dB, {e['n']} paired seeds; "+('interval pending.' if e['low'] is None else f"95% interval {e['low']:+.3f} to {e['high']:+.3f} dB."))
    props=list(dict.fromkeys(e['property'] for e in property_effects if e['material']==material and e['spatiallyVarying']))
    if props:
        lines+=['','Spatially varying property ΔPSNR (complete-chain error, paired mean dB; exploratory):','',
          '| Change | '+' | '.join(props)+' |', '| --- | '+' | '.join('---:' for _ in props)+' |']
        for e in rows:
            values=[]
            for prop in props:
                p=next((p for p in property_effects if p['material']==material and p['variant']==e['variant'] and p['property']==prop),None)
                values.append(f"{p['mean']:+.2f}" if p else 'pending')
            lines.append('| '+labels[e['variant']]+' | '+' | '.join(values)+' |')
        lines+=['']
    # Point-estimate frontier only from complete seed sets, actual transport bytes.
    means=[]
    for v in variants:
        rs=[by.get((material,v,seed)) for seed in seeds]
        if all(rs) and compatible(rs) and all('brotliBytes' in r['storage'] for r in rs):
            means.append(dict(variant=v,bytes=statistics.mean(r['storage']['brotliBytes'] for r in rs),mse=statistics.mean(r['quality']['exported']['normalizedMse'] for r in rs)))
    frontier=[a for a in means if not any(b['bytes']<=a['bytes'] and b['mse']<=a['mse'] and (b['bytes']<a['bytes'] or b['mse']<a['mse']) for b in means)]
    lines+=['','Complete-seed size/error frontier (point estimates, not confidence dominance): '+(', '.join(labels[a['variant']] for a in sorted(frontier,key=lambda a:a['bytes'])) or 'pending')+'.','']
lines+=['**Recommendations for interpreting this screen.** Promote configurations only after all three seeds and source checks complete. Favor smaller configurations whose error meets the application threshold; do not turn a nonsignificant difference into an equivalence claim. Confirm finalists on road, rotate/scale and velvet, then measure rendered quality and decoder time. Keep checkerboard out until its source normal is verified. Extend training on finalists to distinguish capacity from optimization limits.','',
 'The plots show conditional paired deltas, not marginal effects averaged over a full factorial design. Property effects and raw numbers are in `feature-effects.json`; detailed errors and previews remain in the main dashboard.','', '![Paired feature effects](feature-effects.png)']
if failed:lines+=['','Failed cases: '+', '.join(r['id'] for r in failed)]
(out/'findings.md').write_text('\n'.join(lines)+'\n')
(out/'feature-effects.json').write_text(json.dumps(dict(sourceHash=plan['sourceHash'],analysisSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlibVersion=matplotlib.__version__,complete=len(good),expected=len(plan['cases']),effects=effects,propertyEffects=property_effects,interactions=interactions),indent=2)+'\n')
fig,axes=plt.subplots(1,len(materials),figsize=(6*len(materials),8),squeeze=False,layout='constrained')
order=[v for v in labels if v in variants and v!='reference']
for ax,material in zip(axes[0],materials):
    rs={e['variant']:e for e in effects if e['material']==material}
    for i,v in enumerate(order):
        e=rs.get(v)
        if not e:continue
        color='#267a5b' if e['mean']>=0 else '#b95435'
        ax.plot(e['mean'],i,'o',color=color,markersize=5)
        if e['low'] is not None:ax.plot([e['low'],e['high']],[i,i],color=color,lw=1.5)
        ax.annotate(f"n={e['n']}",(e['mean'],i),xytext=(5,6),textcoords='offset points',fontsize=7)
    ax.set_yticks(range(len(order)),[labels[v] for v in order]);ax.invert_yaxis();ax.axvline(0,color='#777',lw=.8)
    ax.set_title(material);ax.set_xlabel('ΔPSNR vs reference (dB); 95% paired CI');ax.grid(axis='x',alpha=.15)
fig.suptitle(f"Controlled feature effects: {len(good)}/{len(plan['cases'])} complete",fontsize=16)
fig.savefig(out/'feature-effects.png',dpi=140);plt.close(fig)
print(f"Findings updated: {len(good)}/{len(plan['cases'])}, {len(effects)} matched effects")
