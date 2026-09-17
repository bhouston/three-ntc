// Declarative design. IDs are stable; changes require a new snapshot/run directory.
export const materials = ['brick','checkerboard_normal','road_aggregate',
  'standard_surface_rotate_scale2d_test','uv_grid','wood','velvet'];
export const reference = {
  gridChannels:4, lowResChannels:4, baseResolution:256, levels:4, mipsPerLevel:2,
  positionalEncoding:true, positionalEncodingPeriod:8, g0Sampling:'concatenate', dualGrid:true,
  hiddenSizes:[32,32], hiddenActivation:'hgelu', batchSize:4096, learningRate:0.01,
  cosineAnnealingScale:0.001, mipFilter:'source', quantization:{mode:'uint4',method:'ste',range:[-0.5,0.4375]},
  retrainAfterQuantize:0.05,
};
export function variants() {
  const result = [{id:'reference', family:'controlled', changes:{}}];
  // Full encoding x sampling x bit-depth crossing: isolates encoding from sampling,
  // and shows whether benefits reverse under aggressive quantization.
  for (const encoding of ['none','cell','tile']) for (const sampling of ['bilinear','concatenate']) for (const bits of [2,4,8]) {
    if (encoding==='tile' && sampling==='concatenate' && bits===4) continue;
    result.push({id:`pe-${encoding}_g0-${sampling}_uint${bits}`,family:'controlled',changes:{
      positionalEncoding:encoding!=='none',positionalEncodingPeriod:encoding==='tile'?8:0,g0Sampling:sampling,
      quantization:{mode:`uint${bits}`,method:'ste',range:[-0.5,0.5-1/(2**bits)]},
    }});
  }
  for(const [id,changes] of Object.entries({
    'activation-relu':{hiddenActivation:'relu'},
    'hidden-16':{hiddenSizes:[16,16]}, 'hidden-64':{hiddenSizes:[64,64]},
    'relu-hidden-16':{hiddenActivation:'relu',hiddenSizes:[16,16]},
    'g1-off':{dualGrid:false},
    'base-128':{baseResolution:128}, 'base-512':{baseResolution:512},
    'levels-2':{levels:2}, 'levels-1':{levels:1},
    'mip-spacing-1':{mipsPerLevel:1}, 'mip-spacing-3':{mipsPerLevel:3},
    'grid-channels-8-12':{gridChannels:8,lowResChannels:12},
  })) result.push({id,family:'controlled',changes});
  for(const profile of ['library','website','paper']) {
    result.push({id:`profile-${profile}-equal-budget`,family:'profile-controlled',profile});
    result.push({id:`profile-${profile}-native`,family:'native',profile});
  }
  return result;
}
export function makeCases({seeds=[1,2,3],suite='all',onlyMaterials=materials,onlyVariants=null,iterations=4000,resolution=1024}={}) {
  const selected=variants().filter(v=>(suite==='all'||(suite==='native'?v.family==='native':v.family!=='native'))&&(!onlyVariants||onlyVariants.includes(v.id)));
  const cases=[];
  for(const seed of seeds) for(const material of onlyMaterials) for(const variant of selected) {
    if(!materials.includes(material)) throw new Error(`Unknown material ${material}`);
    cases.push({id:`${material}__${variant.id}__seed-${seed}`,material,variant,seed,iterations,resolution});
  }
  // A fixed shuffle reduces thermal/order bias. Native runs go last to avoid
  // delaying controlled results behind the much larger paper sample budget.
  let state=937;
  for(let i=cases.length-1;i>0;i--) {state=(Math.imul(state,1664525)+1013904223)>>>0;const j=state%(i+1);[cases[i],cases[j]]=[cases[j],cases[i]];}
  return cases.sort((a,b)=>Number(a.variant.family==='native')-Number(b.variant.family==='native'));
}
