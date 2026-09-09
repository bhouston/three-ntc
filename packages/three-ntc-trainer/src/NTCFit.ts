import { NTCNodeMaterial } from 'three-ntc';
import { CHANNELS, buildChannelActivations } from 'three-ntc';
import { NTCTrainer } from './NTCTrainer.js';
import { bakeMaterialToTextures, classifyMaterialChannels } from './NTCSource.js';

/**
 * End-to-end convenience path covering the sequence a from-scratch consumer
 * of this addon would otherwise have to hand-assemble from five separate
 * low-level pieces: classify the material's channels, bake the active ones
 * to textures, train a `NTCTrainer` against them, and construct (and, on
 * every progress tick, re-construct and dispose the previous)
 * `NTCNodeMaterial`.
 *
 * `options` is passed straight through to `NTCTrainer` (so `levels`,
 * `hiddenSizes`, `iterations`, `learningRate`, etc. all apply), plus a few
 * fit-specific fields: `resolution` (bake resolution, default 512),
 * `debugView` (default 'shaded'), `channels` (the channel vocabulary to fit
 * against, default the built-in `CHANNELS` - see `three-ntc`'s NTCFormat.js),
 * `uvTransform` (a `THREE.Matrix3` mapping mesh/query UV into local space,
 * default identity - see `NTCTextureSource.bakeColorNodeToTexture`'s doc
 * comment and `NTCMaterialXUvTransform.js` for how one gets detected from a
 * MaterialX graph), and `onProgress`, called with the usual `NTCTrainer` progress payload plus a
 * `material` field holding the current (already-disposing-its-predecessor)
 * in-progress material, suitable for live preview during training.
 *
 * Throws if every channel on `material` classifies as constant - see
 * `NTCSource.classifyMaterialChannels` - since there's then nothing for a
 * network to fit; construct directly from a classification's
 * `constantValues` in that case instead.
 */
async function fitNTCMaterial( renderer: any, material: any, options: any = {} ) {

	const { onProgress, resolution = 512, debugView = 'shaded', channels = CHANNELS, uvTransform = null, ...trainerOptions } = options;

	const channelClassification = classifyMaterialChannels( material, channels );

	if ( channelClassification.activeChannels.length === 0 ) {

		throw new Error( 'THREE.NTCFit.fitNTCMaterial: every channel on this material is constant - there is nothing for a network to fit. Use NTCSource.classifyMaterialChannels() directly instead.' );

	}

	// `uvTransform` (see NTCTextureSource.bakeColorNodeToTexture's doc
	// comment) bakes every channel in its local, untransformed space, and is
	// carried onto the trained cpuModel (below, via NTCTrainer's own
	// `uvTransform` option -> NTCGridPyramidModel.js) so `NTCNodeMaterial`
	// maps query UV back into that same space at render time.
	const renderTargets = await bakeMaterialToTextures( renderer, material, resolution, channelClassification.activeChannels, uvTransform );

	const trainer = new NTCTrainer( {
		outputChannels: channelClassification.totalChannels,
		channelActivations: buildChannelActivations( channelClassification.activeChannels ),
		uvTransform,
		...trainerOptions
	} );

	let current: any = null;

	const rebuild = ( cpuModel: any ) => {

		const previous = current;
		current = new NTCNodeMaterial( cpuModel, channelClassification, { debugView, channels } );
		if ( previous ) previous.dispose();

		return current;

	};

	try {

		const result = await trainer.train( {
			renderer,
			sourceTextures: renderTargets.map( ( renderTarget: any ) => renderTarget.texture ),
			onProgress: onProgress ? ( progress: any ) => onProgress( { ...progress, material: rebuild( progress.cpuModel ) } ) : null
		} );

		rebuild( result.cpuModel );

		return {
			material: current,
			channelClassification,
			cpuModel: result.cpuModel,
			loss: result.loss,
			iteration: result.iteration,
			iterations: result.iterations,
			stoppedEarly: result.stoppedEarly
		};

	} finally {

		for ( const renderTarget of renderTargets ) renderTarget.dispose();

	}

}

export { fitNTCMaterial };
