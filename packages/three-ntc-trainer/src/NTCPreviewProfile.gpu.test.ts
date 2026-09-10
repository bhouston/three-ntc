import { AmbientLight, DataTexture, FloatType, HalfFloatType, LinearMipmapLinearFilter, Mesh, OrthographicCamera, PlaneGeometry, RenderTarget, RGBAFormat, Scene } from 'three';
import { WebGPURenderer } from 'three/webgpu';
import { float } from 'three/tsl';
import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { CHANNELS, getChannel, layoutChannels, NTCNodeMaterial } from 'three-ntc';
import { summarizeTimings } from '../../../test/performance-metrics.js';
import { NTCTrainer } from './NTCTrainer.js';
import { readRenderTargetFloats } from '../../../test/gpu-helpers.js';

it('profiles five default-size training steps with two live physical preview updates', async () => {
  // Match the website's separate training and viewing devices and model shape.
  const training = new WebGPURenderer({antialias:false}) as any;
  const viewing = new WebGPURenderer({antialias:false}) as any;
  await Promise.all([training.init(), viewing.init()]);
  const errors:string[]=[];
  const listeners = [training,viewing].map(renderer => {
    const listener=(event:any)=>errors.push(event.error.message);
    renderer.backend.device.addEventListener('uncapturederror',listener);
    renderer.backend.device.lost.then((info:any)=>{if(info.reason!=='destroyed') errors.push(`Device lost: ${info.message}`);});
    return listener;
  });
  const layout=layoutChannels(['albedo','roughness','normal','metalness'].map(key=>getChannel(key)));
  const size=1024;
  const textures=Array.from({length:2},(_,group)=>{
    const data=new Float32Array(size*size*4);
    for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
      const mortar=y%64<4 || (x+(Math.floor(y/64)%2)*64)%128<4;
      const channels=group===0 ? (mortar?[0.5,0.5,0.5,0.9]:[0.45,0.12,0.06,0.7]) : [0,0,0,0];
      data.set(channels,(y*size+x)*4);
    }
    const texture=new DataTexture(data,size,size,RGBAFormat,FloatType);
    texture.generateMipmaps=true;texture.minFilter=LinearMipmapLinearFilter;texture.needsUpdate=true;
    return texture;
  });
  const trainer=new NTCTrainer({gridChannels:4,levels:4,baseResolution:256,hiddenSizes:[32,32],
    positionalEncoding:true,dualGrid:true,outputChannels:7,batchSize:8192,iterations:5,seed:1,
    channelActivations:layout.channels.flatMap(c=>Array(c.size).fill(c.activation))});
  const target=new RenderTarget(256,256,{type:HalfFloatType});
  const geometry=new PlaneGeometry(2,2);geometry.computeTangents();
  const scene=new Scene(),camera=new OrthographicCamera(-1,1,1,-1,0,4);camera.position.z=2;
  scene.add(new AmbientLight(0xffffff,1));
  let material:NTCNodeMaterial|undefined;
  const progress:any[]=[],completions:Promise<void>[]=[];
  let lastHeartbeat=performance.now(),maxHeartbeatGap=0,heartbeats=0;
  const heartbeat=setInterval(()=>{
    const now=performance.now();maxHeartbeatGap=Math.max(maxHeartbeatGap,now-lastHeartbeat);lastHeartbeat=now;heartbeats++;
  },16);
  const device=viewing.backend.device,create=device.createShaderModule.bind(device);
  let shaderModules=0,initialShaderModules=0;
  device.createShaderModule=(d:any)=>{shaderModules++;return create(d);};
  const animationIntervals:number[]=[];
  let lastAnimationFrame=0;
  viewing.setAnimationLoop(()=>{
    if(!material) return;
    const now=performance.now();
    if(lastAnimationFrame) animationIntervals.push(now-lastAnimationFrame);
    lastAnimationFrame=now;
    viewing.setRenderTarget(target);
    viewing.render(scene,camera);
  });
  const start=performance.now();
  try {
    await trainer.train({renderer:training,sourceTextures:textures,onProgress:p=>{
      const frameStart=performance.now();
      if(material) material.updateFromModel(p.cpuModel);
      else {
        material=new NTCNodeMaterial(p.cpuModel,{activeChannels:layout.channels,
          constantValues:Object.fromEntries(CHANNELS.map(c=>[c.key,c.defaultValue]))},{lodNode:float(2.35)});
        scene.add(new Mesh(geometry,material));
      }
      viewing.setRenderTarget(target);viewing.render(scene,camera);
      if(progress.length===0) initialShaderModules=shaderModules;
      const metric={iteration:p.iteration,loss:p.loss,submissionMs:performance.now()-frameStart,completedMs:0};
      progress.push(metric);
      completions.push(device.queue.onSubmittedWorkDone().then(()=>{metric.completedMs=performance.now()-frameStart;}));
    }});
    // End the concurrent preview before readback and isolated frame measurements.
    viewing.setAnimationLoop(null);
    await Promise.all(completions);
    const pixels=await readRenderTargetFloats(viewing,target,256);
    const frames:number[]=[];
    for(let i=0;i<3;i++) {
      const frameStart=performance.now();viewing.render(scene,camera);
      await device.queue.onSubmittedWorkDone();frames.push(performance.now()-frameStart);
    }
    await new Promise(resolve=>setTimeout(resolve,32));
    expect(errors).toEqual([]);
    expect(progress.map(p=>p.iteration)).toEqual([1,5]);
    expect(progress.every(p=>Number.isFinite(p.loss))).toBe(true);
    expect([...pixels].every(Number.isFinite)).toBe(true);
    expect(shaderModules).toBe(initialShaderModules);
    expect(heartbeats).toBeGreaterThan(0);
    expect(animationIntervals.length).toBeGreaterThan(0);
    await (commands as any).recordMetric({kind:'training-preview-profile',userAgent:navigator.userAgent,animationIntervals,animationSummary:animationIntervals.length ? summarizeTimings(animationIntervals) : null,adapter:Object.fromEntries(['vendor','architecture','device','description'].map(key=>[key,device.adapterInfo?.[key]])),
      elapsedMs:performance.now()-start,maxHeartbeatGap,heartbeats,progress,steadyFrameMs:frames,shaderModules,errors});
  } finally {
    viewing.setAnimationLoop(null);
    clearInterval(heartbeat);device.createShaderModule=create;
    [training,viewing].forEach((renderer,i)=>renderer.backend.device.removeEventListener('uncapturederror',listeners[i]));
    material?.dispose();geometry.dispose();target.dispose();textures.forEach(t=>t.dispose());
    training.dispose();viewing.dispose();
  }
});
