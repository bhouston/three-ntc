import { expect, it } from 'vitest';
import { commands } from 'vitest/browser';
import { uv, vec4, float } from 'three/tsl';
import { getChannel, layoutChannels, buildChannelActivations, NTCLoader, applyChannelActivation } from 'three-ntc';
import { getRenderer, renderNodeToFloats } from '../../../test/gpu-helpers.js';
import { NTCTrainer } from './NTCTrainer.js';
import { bakeColorNodeToTexture } from './NTCTextureSource.js';
import { encodeNTC } from './NTCManifest.js';
import { buildLevelTextures, evaluateNeuralTextureRaw } from '../../three-ntc/src/NTCDecoderTSL.js';

it('fits and reloads HDR emission beyond the legacy sigmoid ceiling', async () => {
 const renderer=await getRenderer(); const size=32;
 const source=await bakeColorNodeToTexture(renderer,vec4(uv().x.mul(4).add(2),uv().y.mul(6).add(2),float(4),float(0)),size,{generateMipmaps:true});
 const target=await renderNodeToFloats(renderer,vec4(uv().x.mul(4).add(2),uv().y.mul(6).add(2),float(4),float(0)),size);
 const metrics=[];
 try {
  for (const mode of ['legacy','hdr'] as const) {
   const channel={...getChannel('emissive'), ...(mode==='legacy'?{activation:'sigmoid' as const}:{})};
   const layout=layoutChannels([channel]);
   const options={gridChannels:4,levels:2,baseResolution:8,hiddenSizes:[16,16],hiddenActivation:'hgelu',
    outputChannels:3,channelActivations:buildChannelActivations(layout.channels),batchSize:512,iterations:420,
    seed:7,quantization:{mode:'uint4'},positionalEncoding:true,dualGrid:true};
   const start=performance.now();
   const result=await new NTCTrainer(options).train({renderer,sourceTexture:source.texture});
   const trainingMs=performance.now()-start;
   const classification={activeChannels:layout.channels,constantValues:{}};
   const loaded=new NTCLoader().parse(encodeNTC(result.cpuModel,classification));
   const textures=buildLevelTextures(loaded.cpuModel);
   try {
    const raw=evaluateNeuralTextureRaw(uv(),loaded.cpuModel,null,null,float(0),textures);
    const activation=loaded.channelClassification.activeChannels[0].activation;
    const pixels=await renderNodeToFloats(renderer,vec4(...raw.map(v=>applyChannelActivation(v,activation)),float(0)),size);
    let error=0;for(let i=0;i<pixels.length;i++) if(i%4!==3) error+=(pixels[i]-target[i])**2;
    metrics.push({mode,activation,mse:error/(size*size*3),trainingMs});
   } finally { textures.forEach(t=>t.dispose()); }
  }
  expect(metrics[1].mse).toBeLessThan(metrics[0].mse*0.1);
  await commands.recordMetric({kind:'hdr-emission',size,iterations:420,batchSize:512,seed:7,metrics});
 } finally { source.dispose(); }
});
