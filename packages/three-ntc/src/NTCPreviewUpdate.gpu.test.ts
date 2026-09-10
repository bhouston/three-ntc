import { Mesh, OrthographicCamera, PlaneGeometry, RenderTarget, Scene, HalfFloatType } from 'three';
import { float } from 'three/tsl';
import { expect, it } from 'vitest';
import { NTCNodeMaterial } from './NTCNodeMaterial.js';
import { CHANNELS, getChannel, layoutChannels } from './NTCFormat.js';
import { getRenderer, makeModel, readRenderTargetFloats } from '../../../test/gpu-helpers.js';

it('uploads preview weights and latents without rebuilding its shader or textures',async()=>{
  const renderer=await getRenderer();
  const model=makeModel(1,{gridChannels:1,levels:1,baseResolution:4,textureResolution:16,
    hiddenSizes:[],outputChannels:3});
  const layer=model.decoder.layers[0];layer.weights.fill(0);layer.biases.fill(0);
  for(let c=0;c<3;c++) layer.weights[c*layer.inputSize]=1;
  model.grids[0].data.fill(0.2);
  const material=new NTCNodeMaterial(model,{activeChannels:layoutChannels([getChannel('albedo')]).channels,
    constantValues:Object.fromEntries(CHANNELS.map(c=>[c.key,c.defaultValue]))},{debugView:'albedo',lodNode:float(0)});
  const geometry=new PlaneGeometry(2,2);
  const scene=new Scene();scene.add(new Mesh(geometry,material));
  const camera=new OrthographicCamera(-1,1,1,-1,0,4);camera.position.z=2;
  const target=new RenderTarget(16,16,{type:HalfFloatType});
  const device=renderer.backend.device,create=device.createShaderModule.bind(device);
  let compilations=0;
  device.createShaderModule=(d:any)=>{compilations++;return create(d);};
  const previous=renderer.getRenderTarget();
  try {
    renderer.setRenderTarget(target);await renderer.compileAsync(scene,camera);
    renderer.render(scene,camera);
    const before=await readRenderTargetFloats(renderer,target,16);
    const initialCompilations=compilations,version=material.version;
    const textures=material.levelTextures!.slice();
    model.grids[0].data.fill(0.8);
    for(let c=0;c<3;c++) layer.weights[c*layer.inputSize]=2;
    layer.biases.fill(0.4);
    material.updateFromModel(model);
    renderer.render(scene,camera);
    const after=await readRenderTargetFloats(renderer,target,16);
    expect(compilations).toBe(initialCompilations);
    expect(material.version).toBe(version);
    expect(material.levelTextures).toEqual(textures);
    for(let i=0;i<after.length;i+=4) for(let c=0;c<3;c++) {
      expect(Math.abs(before[i+c]-1/(1+Math.exp(-0.2)))).toBeLessThan(0.002);
      expect(Math.abs(after[i+c]-1/(1+Math.exp(-2)))).toBeLessThan(0.002);
    }
    model.grids[0].width=8;
    expect(()=>material.updateFromModel(model)).toThrow(/shape changed/);
  } finally {
    device.createShaderModule=create;renderer.setRenderTarget(previous);
    target.dispose();geometry.dispose();material.dispose();
  }
});
