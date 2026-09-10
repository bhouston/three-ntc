import { AmbientLight, Mesh, OrthographicCamera, PlaneGeometry, RenderTarget, Scene, HalfFloatType } from 'three';
import { float, vec2, vec4 } from 'three/tsl';
import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { NTCNodeMaterial } from './NTCNodeMaterial.js';
import { CHANNELS, getChannel, layoutChannels } from './NTCFormat.js';
import { evaluateNeuralTextureFiltered } from './NTCDecoderTSL.js';
import { buildLevelTextures } from './NTCHalfFloatTexture.js';
import { getRenderer, makeModel, evaluateNTCCpu, renderNodeToFloats, readRenderTargetFloats } from '../../../test/gpu-helpers.js';

// Conservative WGSL private-storage accounting: round every vector and struct
// member up to 16 bytes. Unknown types fail the test instead of undercounting.
function privateStorageUpperBound(code:string):number {
  const structs=new Map([...code.matchAll(/struct\s+(\w+)\s*\{([^}]+)\}/g)].map(m=>[m[1],m[2]]));
  const size=(type:string):number=>{
    type=type.trim().replace(/[,;]$/,'').replace(/\s/g,'');
    if(/^(f32|i32|u32|bool)$/.test(type)) return 4;
    if(/^vec[234]<[fiu]32>$/.test(type)) return 16;
    const matrix=type.match(/^mat([234])x[234]<f32>$/);if(matrix) return Number(matrix[1])*16;
    const array=type.match(/^array<(.+),\s*(\d+)u?>$/);if(array) return size(array[1])*Number(array[2]);
    const body=structs.get(type);
    if(body) return [...body.matchAll(/:\s*([^\n]+)/g)].reduce((sum,m)=>sum+Math.ceil(size(m[1])/16)*16,0);
    throw new Error(`Unaccounted WGSL type: ${type}`);
  };
  return [...code.matchAll(/var<private>\s+\w+\s*:\s*([^;]+);/g)].reduce((sum,m)=>sum+size(m[1]),0);
}

for(const hiddenSize of [32,64]) it(`keeps the ${hiddenSize}-wide full physical material below the 8 KB private-storage limit`,async()=>{
  const times:Record<string,number>={}; let start=performance.now();
  const renderer=await getRenderer(); times.init=performance.now()-start;
  const layout=layoutChannels(['albedo','roughness','normal','metalness'].map(k=>getChannel(k)));
  const model=makeModel(7,{gridChannels:4,levels:4,baseResolution:256,textureResolution:1024,
    positionalEncoding:true,dualGrid:true,hiddenSizes:[hiddenSize,hiddenSize],outputChannels:layout.totalChannels});
  const material=new NTCNodeMaterial(model,{activeChannels:layout.channels,constantValues:Object.fromEntries(CHANNELS.map(c=>[c.key,c.defaultValue]))},{lodNode:float(2.35)});
  const geometry=new PlaneGeometry(2,2);geometry.computeTangents();
  const scene=new Scene();scene.add(new Mesh(geometry,material),new AmbientLight(0xffffff,1));
  const camera=new OrthographicCamera(-1,1,1,-1,0,4);camera.position.z=2;
  const target=new RenderTarget(16,16,{type:HalfFloatType});
  const device=renderer.backend.device;
  const create=device.createShaderModule.bind(device);
  const shaders:string[]=[];
  device.createShaderModule=(descriptor:any)=>{shaders.push(descriptor.code);return create(descriptor);};
  const previous=renderer.getRenderTarget();
  let bytes=0;
  try {
    renderer.setRenderTarget(target);
    start=performance.now(); await renderer.compileAsync(scene,camera); times.compile=performance.now()-start; start=performance.now();
    renderer.render(scene,camera);
    const pixels=await readRenderTargetFloats(renderer,target,16);
    times.renderRead=performance.now()-start;
    expect([...pixels].every(Number.isFinite)).toBe(true);
    const fragment=shaders.find(code=>code.includes('@fragment'));
    expect(fragment).toBeDefined();
    // One nested matrix loop per layer, independent of network width.
    expect((fragment!.match(/for \( var ntcInput /g) ?? []).length).toBe(model.decoder.layers.length);
    expect((fragment!.match(/for \( var ntcOutput /g) ?? []).length).toBe(model.decoder.layers.length);
    bytes=privateStorageUpperBound(fragment!);
    expect(bytes).toBeLessThanOrEqual(8192);
  } finally {
    device.createShaderModule=create;renderer.setRenderTarget(previous);
    target.dispose();geometry.dispose();material.dispose();
  }
  // Independent CPU decode-then-filter oracle, including a fractional mip.
  const u=0.413,v=0.277,lod=2.35;
  const expected=Array(model.outputChannels).fill(0);
  for(let m=0;m<2;m++) {
    const level=Math.floor(lod)+m,size=1024>>level;
    const px=u*size-0.5,py=v*size-0.5,tx=px-Math.floor(px),ty=py-Math.floor(py);
    for(let y=0;y<2;y++) for(let x=0;x<2;x++) {
      const values=evaluateNTCCpu(model,(Math.floor(px)+x+0.5)/size,(Math.floor(py)+y+0.5)/size,level);
      const weight=(x?tx:1-tx)*(y?ty:1-ty)*(m?lod%1:1-lod%1);
      for(let c=0;c<expected.length;c++) expected[c]+=values[c]*weight;
    }
  }
  start=performance.now();
  const textures=buildLevelTextures(model);
  const values=evaluateNeuralTextureFiltered(vec2(u,v),model,textures,float(lod));
  let squaredError=0;
  for(let group=0;group<Math.ceil(expected.length/4);group++) {
    const lanes=Array.from({length:4},(_,i)=>values[group*4+i] ?? float(0));
    const actual=await renderNodeToFloats(renderer,vec4(...lanes),1);
    for(let i=0;i<4 && group*4+i<expected.length;i++) {
      const error=actual[i]-expected[group*4+i];
      expect(Math.abs(error)).toBeLessThan(0.002);
      squaredError+=error*error;
    }
  }
  textures.forEach(t=>t.dispose());
  times.oracleRenders=performance.now()-start;
  await (commands as any).recordMetric({times,kind:'runtime-shader',hiddenSize,privateBytesUpperBound:bytes,mse:squaredError/expected.length});
});
