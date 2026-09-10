import { AmbientLight, EquirectangularReflectionMapping, DirectionalLight, HalfFloatType, Mesh, PerspectiveCamera, RenderTarget, Scene, SphereGeometry } from 'three';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import hdrUrl from '../../website/public/textures/equirectangular/san_giuseppe_bridge_2k.hdr?url';
import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { float, uv, vec4 } from 'three/tsl';
import { evaluateNeuralTextureRaw } from './NTCDecoderTSL.js';
import { NTCNodeMaterial } from './NTCNodeMaterial.js';
import { NTCLoader } from './NTCLoader.js';
import { getRenderer, readRenderTargetFloats, evaluateNTCCpu, pixelUv, renderNodeToFloats } from '../../../test/gpu-helpers.js';
import brick from '../../website/public/ntc/brick.ntc?raw';

it('profiles the shipped brick in a standalone 1024px viewer without training',async()=>{
  const renderer=await getRenderer();
  const {cpuModel,channelClassification}=new NTCLoader().parse(brick);
  const material=new NTCNodeMaterial(cpuModel,channelClassification);
  const geometry=new SphereGeometry(1,64,64);geometry.computeTangents();
  const scene=new Scene();scene.add(new Mesh(geometry,material),new AmbientLight(0xffffff,0.6));
  const key=new DirectionalLight(0xffffff,2.5);key.position.set(3,4,5);scene.add(key);
  const fill=new DirectionalLight(0xffffff,1);fill.position.set(-4,-2,-3);scene.add(fill);
  const environment=await new HDRLoader().loadAsync(hdrUrl);
  environment.mapping=EquirectangularReflectionMapping;scene.environment=environment;
  const camera=new PerspectiveCamera(40,1,0.1,100);camera.position.z=3.2;
  const target=new RenderTarget(1024,1024,{type:HalfFloatType});
  const device=renderer.backend.device,previous=renderer.getRenderTarget();
  const errors:string[]=[];
  const listener=(e:any)=>errors.push(e.error.message);device.addEventListener('uncapturederror',listener);
  let alive=true;device.lost.then((info:any)=>{if(alive) errors.push(`Device lost: ${info.message}`);});
  try {
    renderer.setRenderTarget(target);
    const times:number[]=[];
    for(let i=0;i<5;i++) {
      const start=performance.now();renderer.render(scene,camera);
      await device.queue.onSubmittedWorkDone();times.push(performance.now()-start);
    }
    const pixels=await readRenderTargetFloats(renderer,target,1024);
    expect([...pixels].every(Number.isFinite)).toBe(true);
    expect(errors).toEqual([]);
    const raw=evaluateNeuralTextureRaw(uv(),cpuModel,null,null,float(0),material.levelTextures);
    let squaredError=0,observations=0;
    for(let group=0;group<Math.ceil(cpuModel.outputChannels/4);group++) {
      const values=await renderNodeToFloats(renderer,vec4(...Array.from({length:4},(_,i)=>raw[group*4+i] ?? float(0))),16);
      for(let y=0;y<16;y++) for(let x=0;x<16;x++) {
        const [u,v]=pixelUv(16,x,y),expected=evaluateNTCCpu(cpuModel,u,v,0);
        for(let c=0;c<4 && group*4+c<expected.length;c++) {
          const error=values[(y*16+x)*4+c]-expected[group*4+c];
          expect(Math.abs(error)).toBeLessThan(0.05);
          squaredError+=error*error;observations++;
        }
      }
    }
    await (commands as any).recordMetric({kind:'brick-viewer',adapter:Object.fromEntries(
      ['vendor','architecture','device','description'].map(k=>[k,device.adapterInfo?.[k]])),
      rawCpuMse:squaredError/observations,model:cpuModel.decoder.layers.map(l=>[l.inputSize,l.outputSize]),firstFrameMs:times[0],steadyFrameMs:times.slice(1),errors});
  } finally {
    alive=false;device.removeEventListener('uncapturederror',listener);renderer.setRenderTarget(previous);
    target.dispose();material.dispose();geometry.dispose();environment.dispose();
  }
});
