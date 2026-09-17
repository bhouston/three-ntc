import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { Matrix3 } from 'three';
import { float, textureLevel, uv, vec4 } from 'three/tsl';
import { NTCLoader, buildChannelActivations, applyChannelActivation } from 'three-ntc';
import { MaterialXLoader, classifyMaterialChannels, bakeMaterialToTextures, inferAlbedoUvTransform, NTCTrainer, getNTCPaperProfile } from 'three-ntc-trainer';
import { encodeNTC } from '../../packages/three-ntc-trainer/src/NTCManifest.js';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../../packages/three-ntc/src/NTCDecoderTSL.js';
import { getRenderer, renderNodeToFloats } from '../../test/gpu-helpers.js';
import { WEBSITE_TRAINING_DEFAULTS as web } from '../../packages/website/src/lib/training-defaults.js';
import { reference } from './design.mjs';
import { measure, summarizeMips, normalError } from './metrics.mjs';

const api=commands as any;
it('runs one reproducible MaterialX benchmark case', async()=>{
  const config=await api.ntcBenchmarkCase();
  const renderer=await getRenderer();
  const device=renderer.backend.device;
  if (!device) throw new Error('WebGPU device unavailable; Three.js selected '+renderer.backend.constructor.name+
    '. Run node benchmarks/ntc/diagnose-gpu.mjs. Details: '+JSON.stringify({
      secureContext:isSecureContext,gpuApi:!!navigator.gpu,userAgent:navigator.userAgent}));
  const errors:string[]=[];
  const listener=(event:any)=>errors.push(event.error.message);
  device.addEventListener('uncapturederror',listener);
  let active=true;device.lost.then((info:any)=>{if(active)errors.push(`Device lost: ${info.message}`);});
  const resources:any[]=[];
  try {
    const asset:any=await new MaterialXLoader().loadAsync(`/packages/website/public/materialx/${config.material}.mtlx`,{uvSpace:'top-left',throwOnErrors:true});
    if(asset.texturesReady)await asset.texturesReady;
    // `asset.errors` is captured at parse time; asynchronous image failures are
    // appended later to the live log. Never measure an unloaded black texture.
    const sourceErrors=(asset.log??[]).filter((entry:any)=>entry.severity==='error');
    if(sourceErrors.length)throw new Error('MaterialX source failed: '+sourceErrors.map((entry:any)=>entry.message).join('; '));
    const source:any=Object.values(asset.materials ?? asset)[0];resources.push(source);
    const classification=classifyMaterialChannels(source);
    if(!classification.totalChannels)throw new Error('Material has no active channels');
    const transform=source.materialXDocument && source.materialXSurfaceShaderNode
      ? inferAlbedoUvTransform(source.materialXDocument,source.materialXSurfaceShaderNode):new Matrix3();
    const targets=await bakeMaterialToTextures(renderer,source,config.resolution,classification.activeChannels,transform);
    resources.push(...targets);
    const activations=buildChannelActivations(classification.activeChannels);
    const c=classification.totalChannels;
    const ranges=activations.map(a=>a==='tanh'?2:1);
    const unpack=async(node:any[],size:number)=>{
      const packed=new Float32Array(size*size*c);
      for(let group=0;group<Math.ceil(c/4);group++) {
        const data=await renderNodeToFloats(renderer,vec4(...Array.from({length:4},(_,k)=>node[group*4+k] ?? float(0))),size);
        for(let p=0;p<size*size;p++)for(let k=0;k<4&&group*4+k<c;k++)packed[p*c+group*4+k]=data[p*4+k];
      }
      return packed;
    };
    // Complete source mip chain, not stochastic training-batch loss. The same
    // box-filtered source reference is used for every variant, including paper.
    const refs:Float32Array[]=[];
    for(let lod=0;lod<=Math.log2(config.resolution);lod++) {
      const nodes:any[]=[];for(const target of targets){const t=textureLevel(target.texture,uv(),lod);nodes.push(t.x,t.y,t.z,t.w);}
      refs.push(await unpack(nodes,Math.max(1,config.resolution>>lod)));
    }
    const referenceHashes=await Promise.all(refs.map(async data=>
      [...new Uint8Array(await crypto.subtle.digest('SHA-256',data.buffer))].map(b=>b.toString(16).padStart(2,'0')).join('')));
    const sourceAudit=classification.activeChannels.map((ch:any)=>{
      const components=Array.from({length:ch.size},(_,k)=>{
        let min=Infinity,max=-Infinity;
        for(let i=ch.offset+k;i<refs[0].length;i+=c){min=Math.min(min,refs[0][i]);max=Math.max(max,refs[0][i]);}
        if(!Number.isFinite(min)||!Number.isFinite(max))throw new Error('Non-finite baked source');
        return {min,max};
      });
      return {key:ch.key,components,spatiallyVarying:components.some(v=>v.max>v.min)};
    });
    if(!sourceAudit.some((ch:any)=>ch.spatiallyVarying))throw new Error('Benchmark source is entirely constant; inspect MaterialX graph and bake before training');
    let requested:any;
    if(config.variant.profile==='library')requested={};
    else if(config.variant.profile==='website')requested={gridChannels:4,levels:web.levels,baseResolution:web.baseResolution,
      hiddenSizes:Array(web.hiddenLayers).fill(web.hiddenSize),hiddenActivation:web.hiddenActivation,
      positionalEncoding:web.positionalEncoding,dualGrid:web.dualGrid,batchSize:web.batchSize,
      learningRate:web.learningRate,quantization:{mode:web.quantization}};
    else if(config.variant.profile==='paper')requested=getNTCPaperProfile(config.resolution);
    else requested={...reference,...config.variant.changes};
    // Equal-budget profile comparisons retain each profile's architecture,
    // optimizer settings, and quantization behavior; only batch/iterations match.
    if(config.variant.family==='profile-controlled')requested={...requested,batchSize:4096};
    requested={...requested,iterations:config.iterations,seed:config.seed,textureResolution:config.resolution,
      outputChannels:classification.totalChannels,channelActivations:activations,uvTransform:transform,
      progressInterval:config.variant.family==='native'?4:128};
    const logicalBatch=requested.batchSize ?? 4096;
    if(logicalBatch>8192)requested.microBatchSize=4096;
    const trainer=new NTCTrainer(requested);
    const lossCurve:any[]=[];let precision:string|undefined;
    const start=performance.now();
    const result=await trainer.train({renderer,sourceTextures:targets.map(t=>t.texture),onProgress:(p:any)=>{
      if(!Number.isFinite(p.loss))throw new Error('Non-finite training loss');
      precision=p.gpuModel.floatGradients?'float':'fixed';
      lossCurve.push({iteration:p.iteration,loss:p.loss,elapsedMs:performance.now()-start});
      void api.ntcBenchmarkProgress({id:config.id,iteration:p.iteration,iterations:config.iterations,elapsedMs:performance.now()-start});
    }});
    await device.queue.onSubmittedWorkDone();
    const trainingMs=performance.now()-start;
    expect(result.iteration).toBe(config.iterations);expect(result.stoppedEarly).toBe(false);
    const manifest=encodeNTC(result.cpuModel,classification);
    const loaded=new NTCLoader().parse(manifest).cpuModel;
    const quality:any={};let preview:any;
    const evaluationStart=performance.now();
    for(const [label,model] of [['trained',result.cpuModel],['exported',loaded]] as const) {
      const textures=buildLevelTextures(model);
      try {
        const mips=[];
        for(let lod=0;lod<=model.maxLod;lod++) {
          const size=Math.max(1,config.resolution>>lod);
          const raw=evaluateNeuralTextureRaw(uv(),model,null,null,float(lod),textures);
          const decoded=await unpack(raw.map((v,i)=>applyChannelActivation(v,activations[i])),size);
          const measured=measure(refs[lod],decoded,size,ranges);
          measured.normals=Object.fromEntries(classification.activeChannels.filter((ch:any)=>ch.type==='normal')
            .map((ch:any)=>[ch.key,normalError(refs[lod],decoded,c,ch.offset)]));
          mips.push(measured);
          if(label==='exported'&&lod===0) {
            const variation=(ch:any)=>Array.from({length:ch.size},(_,k)=>{
              const q=measured.perComponent[ch.offset+k];return (q.referenceMax-q.referenceMin)/ranges[ch.offset+k];
            }).reduce((a:number,b:number)=>a+b,0)/ch.size;
            const color=[...classification.activeChannels].sort((a:any,b:any)=>variation(b)-variation(a))[0];
            const canvas=document.createElement('canvas');canvas.width=768;canvas.height=256;
            const ctx=canvas.getContext('2d')!,image=ctx.createImageData(768,256);
            for(let y=0;y<256;y++)for(let x=0;x<256;x++)for(let k=0;k<3;k++) {
              const index=(Math.floor(y*size/256)*size+Math.floor(x*size/256))*c+color.offset+Math.min(k,color.size-1);
              const a=refs[0][index],b=decoded[index];
              for(let panel=0;panel<3;panel++) {
                const display=(v:number)=>color.activation==='tanh'?(v+1)/2:v;
                const value=panel===0?display(a):panel===1?display(b):Math.abs(a-b)*10;
                const out=(y*768+panel*256+x)*4;image.data[out+k]=Math.round(Math.min(1,Math.max(0,value))*255);image.data[out+3]=255;
              }
            }
            ctx.putImageData(image,0,0);preview={channel:color.key,dataUrl:canvas.toDataURL('image/png')};
          }
        }
        quality[label]=summarizeMips(mips);
      } finally {textures.forEach(t=>t.dispose());}
    }
    await device.queue.onSubmittedWorkDone();
    expect(errors).toEqual([]);
    const grids=[...loaded.grids,...(loaded.lowResGrids??[])];
    const parameters=loaded.decoder.layers.reduce((s,l)=>s+l.weights.length+l.biases.length,0);
    const info=device.adapterInfo;
    await api.ntcBenchmarkRecord({schemaVersion:1,...config,status:'complete',
      adapter:Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info?.[k]??'unknown'])),
      gradientPrecision:precision,options:{...trainer.options,uvTransform:transform.toArray()},
      referenceHashes,completedIterations:result.iteration,samples:logicalBatch*result.iteration,trainingMs,
      iterationsPerSecond:result.iteration/(trainingMs/1000),evaluationMs:performance.now()-evaluationStart,
      channels:classification.activeChannels.map(({key,size,offset,type,activation}:any)=>({key,size,offset,type,activation})),
      constantValues:classification.constantValues,sourceLog:asset.log,sourceAudit,lossCurve,quality,preview,manifest,
      shape:{g0:loaded.grids.map(g=>[g.width,g.height,g.channels]),g1:loaded.lowResGrids?.map(g=>[g.width,g.height,g.channels])??[],decoder:loaded.decoder.layers.map(l=>[l.inputSize,l.outputSize,l.activation])},
      storage:{jsonBytes:new TextEncoder().encode(JSON.stringify(manifest)).length,
        latentPayloadBytes:[...manifest.latents.levels,...(manifest.latents.lowResLevels??[])].reduce((s,g)=>s+atob(g.dataBase64).length,0),
        runtimeGridBytes:grids.reduce((s,g)=>s+g.width*g.height*Math.ceil(g.channels/4)*4*2,0),decoderParameters:parameters},errors});
  } finally {active=false;device.removeEventListener('uncapturederror',listener);resources.forEach(r=>r.dispose?.());}
}, 24*60*60*1000);
