import {
	packVec4Inputs,
	unpackVec4Outputs,
	packLayerWeightsMat4,
	packLayerBiasesVec4,
	evaluateLinearLayerMat4,
	createMat4Storage,
	createVec4Storage
} from './NTCMLPTSL.js';
import { buildMipChainTexture, buildLevelTextures, NTCGrid } from './NTCHalfFloatTexture.js';
import { selectFeatureLevelTSL } from './NTCMipBands.js';
import { computeTiledPositionalEncodingTSL } from './NTCPositionalEncodingTSL.js';
import { applyChannelActivation, NTCActivation } from './NTCOutputActivations.js';
import { MLPLayer } from './NTCBinaryCodec.js';
import { Fn, Loop, array, float, floor, int, pow, textureLevel, vec2, vec3 } from 'three/tsl';

/**
 * A fully decoded `.ntc` CPU model - what `NTCLoader.parse()` produces and
 * `NTCNodeMaterial`'s constructor / `evaluateNeuralTextureRaw` consume.
 */
export interface NTCCpuModel {
	channels: number;
	levels: number;
	mipsPerLevel: number;
	maxLod: number;
	textureResolution?: number;
	positionalEncodingPeriod?: number;
	lodOffset?: number;
	grids: NTCGrid[];
	lowResGrids?: NTCGrid[];
	decoder: { layers: MLPLayer[] };
	outputChannels: number;
	wrap: string;
	uvTransform: any;
	// Optional (default false/absent) - see NTCGridPyramidModel.js (trainer)
	// `computeDecoderInputSize` doc comment and `evaluateNeuralTextureRaw`
	// below.
	positionalEncoding?: boolean;
	// Optional (default false/absent) - the paper's G0/G1 pair, see
	// `computeDecoderInputSize` and `sampleFeatures` below.
	dualGrid?: boolean;
}

/** Reads an individual feature vector from an RGBA array texture. Each layer
 * holds four channels; interpolation is performed after decoding stored values.
 */
function gridFeatures(uv: any, grid: NTCGrid, texture: any, concatenate: boolean): {values:any[];tx:any;ty:any} {
	const x=uv.x.mul(grid.width).sub(0.5), y=uv.y.mul(grid.height).sub(0.5);
	const x0=floor(x), y0=floor(y), tx=x.sub(x0), ty=y.sub(y0);
	const weights=[float(1).sub(tx).mul(float(1).sub(ty)),tx.mul(float(1).sub(ty)),float(1).sub(tx).mul(ty),tx.mul(ty)];
	const values:any[]=Array.from({length:concatenate ? 4*grid.channels : grid.channels},()=>float(0));
	for(let t=0;t<4;t++) {
		const coord=vec2(x0.add(t%2+0.5).div(grid.width),y0.add(Math.floor(t/2)+0.5).div(grid.height));
		for(let group=0;group<Math.ceil(grid.channels/4);group++) {
			const sample=textureLevel(texture,coord,0).depth(int(group));
			const components=[sample.x,sample.y,sample.z,sample.w];
			for(let k=0;k<4 && group*4+k<grid.channels;k++) {
				const c=group*4+k;
				if(concatenate) values[t*grid.channels+c]=components[k];
				else values[c]=values[c].add(components[k].mul(weights[t]));
			}
		}
	}
	return {values,tx,ty};
}

function sampleFeatures(uv: any, model: NTCCpuModel, textures: any[], lod: any): any[] {
	const selected=selectFeatureLevelTSL(lod,model.grids.length,model.mipsPerLevel,model.lodOffset);
	const highWidth=model.positionalEncoding ? 4*model.channels+12 : model.channels;
	const lowChannels=model.lowResGrids?.[0]?.channels ?? model.channels;
	const features:any[]=Array.from({length:highWidth+(model.dualGrid?lowChannels:0)},()=>float(0));
	for(let g=0;g<model.grids.length;g++) {
		const gate=selected.equal(int(g)).select(1,0);
		const high=gridFeatures(uv,model.grids[g],textures[g],!!model.positionalEncoding);
		if(model.positionalEncoding) {
			const size = floor(float(model.textureResolution ?? 2 ** model.maxLod).div(pow(2,lod))).max(1);
			const phase = uv.mul(size).div(model.positionalEncodingPeriod ?? 1);
			high.values.push(...computeTiledPositionalEncodingTSL(
				model.positionalEncodingPeriod ? phase.x : high.tx,
				model.positionalEncodingPeriod ? phase.y : high.ty));
		}
		for(let c=0;c<highWidth;c++) features[c]=features[c].add(high.values[c].mul(gate));
		if(model.dualGrid) {
			const lowIndex=model.lowResGrids?.length ? model.grids.length+g : model.grids.length-1;
			const grid=model.lowResGrids?.[g] ?? model.grids[model.grids.length-1];
			const low=gridFeatures(uv,grid,textures[lowIndex],false);
			for(let c=0;c<lowChannels;c++) features[highWidth+c]=features[highWidth+c].add(low.values[c].mul(gate));
		}
	}
	return features;
}

/** Evaluates native stored features at a requested LOD. Latents must never
 * be downsampled or blended between feature levels before a nonlinear decoder.
 */
function evaluateNeuralTextureRaw( uvNode: any, cpuModel: NTCCpuModel, mipChainTexture: any, renderer: any | null = null, lodNode: any = null, levelTextures: any[] | null = null, packed?: ReturnType<typeof packDecoder> ): any[] {

	// renderer is accepted for compatibility with the higher-level material
	// constructor, but this stock-Three.js path always uses fp32 uniforms.
	void renderer;
	void mipChainTexture;

	const resolvedLodNode = lodNode || float( 0 );
	if (!levelTextures) throw new Error('NTC decoder requires native levelTextures from buildLevelTextures().');
	const features=sampleFeatures(uvNode,cpuModel,levelTextures,resolvedLodNode);

	// Append the normalized LOD value as the decoder's final input component
	// - must match NTCGridPyramidModel.js's `computeDecoderInputSize` /
	// NTCGPUComputeTSL.js's forward pass exactly.
	// Math.max(1, ...) guards against a genuine maxLod of 0 (a model that
	// only ever supports LOD 0, see NTCGridPyramidModel.js) - dividing by 0
	// there would produce a NaN feature, matching the same guard
	// NTCGPUComputeTSL.js's training kernel already applies.
	features.push( resolvedLodNode.div( Math.max( 1, cpuModel.maxLod ) ) );

	// Keep decoder inputs materialized before the dense-layer shader loops.
	// Each layer uses mat4/vec4 blocks with runtime loop bounds to avoid a
	// large statically expanded network in the eight-tap reconstruction loop.
	let activations = packVec4Inputs( features ).map(value => value.toVar());

	const parameters = packed ?? packDecoder(cpuModel);
	for ( let l = 0; l < cpuModel.decoder.layers.length; l ++ ) {

		const layer = cpuModel.decoder.layers[ l ];
		const weights = parameters[l].weights;
		const biases = parameters[l].biases;
		const inputVectorCount = Math.ceil( layer.inputSize / 4 );

		activations = evaluateLinearLayerMat4(
			activations, layer.inputSize, layer.outputSize, layer.activation,
			( outputVector: any, inputVector: any ) => weights.node.element( int(outputVector).mul(inputVectorCount).add(inputVector) ),
			( outputVector: any ) => biases.node.element( outputVector )
		);

	}

	const lastLayer = cpuModel.decoder.layers[ cpuModel.decoder.layers.length - 1 ];

	return unpackVec4Outputs( activations, lastLayer.outputSize );

}

function packDecoder(model: NTCCpuModel) {
	return model.decoder.layers.map(layer => ({
		weights: createMat4Storage(packLayerWeightsMat4(layer.weights, layer.inputSize, layer.outputSize)),
		biases: createVec4Storage(packLayerBiasesVec4(layer.biases))
	}));
}

/** Sampling is selected in JavaScript when building the TSL graph. */
export const NTC_SAMPLING_MODES = ['nearest', 'stochastic', 'trilinear'] as const;
export type NTCSamplingMode = typeof NTC_SAMPLING_MODES[number];

/** Reconstruct one texel for nearest/stochastic, or eight for explicit trilinear.
 * Random components must be independent uniform samples in [0,1). Stochastic
 * sampling selects a physical mip and texel; it does not blend latent features.
 */
function evaluateNeuralTextureSampled(uv: any, model: NTCCpuModel, textures: any[], lod: any,
 activations: NTCActivation[] = [], samplingMode: NTCSamplingMode = 'nearest', random: any = vec3(0.5), packed = packDecoder(model)): any[] {
 if (!NTC_SAMPLING_MODES.includes(samplingMode)) throw new Error(`Unknown NTC sampling mode: ${samplingMode}`);
 const sampled = Fn(() => {
  const resolved = lod.clamp(0, model.maxLod).toVar();
  // These are JavaScript branches: only the selected path enters the node graph.
  if (samplingMode !== 'trilinear') {
   const stochastic = samplingMode === 'stochastic';
   const level = floor(resolved.add(stochastic ? random.z : 0.5)).toVar();
   const size = floor(float(model.textureResolution ?? 2 ** model.maxLod).div(pow(2,level))).max(1).toVar();
   const pixel = stochastic ? uv.mul(size).add(random.xy.sub(0.5)) : uv.mul(size);
   const center = floor(pixel).add(0.5).div(size).fract();
   const raw = evaluateNeuralTextureRaw(center,model,null,null,level,textures,packed);
   return array(raw.map((value,c) => applyChannelActivation(value,activations[c])));
  }
  const result = array(Array.from({length:model.outputChannels}, () => float(0))).toVar();
  const lower = floor(resolved).toVar(), upper = lower.add(1).min(model.maxLod).toVar();
  const blend = resolved.sub(lower).toVar();
  Loop({start:int(0), end:int(8), type:'int', condition:'<', name:'ntcTrilinearTap'}, ({ntcTrilinearTap:i}: {ntcTrilinearTap:any}) => {
   const firstMip = i.lessThan(4).toVar();
   const level = firstMip.select(lower, upper).toVar();
   const x = i.mod(2), y = i.div(2).mod(2);
   const size = floor(float(model.textureResolution ?? 2 ** model.maxLod).div(pow(2,level))).max(1).toVar();
   const pixel = uv.mul(size).sub(0.5).toVar();
   const base = floor(pixel).toVar();
   const fraction = pixel.sub(base).toVar();
   const center = base.add(vec2(float(x).add(0.5),float(y).add(0.5))).div(size).fract();
   const raw = evaluateNeuralTextureRaw(center,model,null,null,level,textures,packed);
   const weight = x.equal(1).select(fraction.x,fraction.x.oneMinus())
    .mul(y.equal(1).select(fraction.y,fraction.y.oneMinus()))
    .mul(firstMip.select(blend.oneMinus(), blend));
   for(let c=0;c<model.outputChannels;c++) result.element(c).addAssign(applyChannelActivation(raw[c],activations[c]).mul(weight));
  });
  return result;
 })();
 return Array.from({length:model.outputChannels}, (_,c) => sampled.element(c));
}

/** Explicit eight-decode trilinear reference, retained for filtering diagnostics. */
function evaluateNeuralTextureFiltered(uv: any, model: NTCCpuModel, textures: any[], lod: any,
 activations: NTCActivation[] = []): any[] {
 return evaluateNeuralTextureSampled(uv,model,textures,lod,activations,'trilinear');
}

export { packDecoder, evaluateNeuralTextureSampled, evaluateNeuralTextureFiltered, evaluateNeuralTextureRaw, buildMipChainTexture, buildLevelTextures };
