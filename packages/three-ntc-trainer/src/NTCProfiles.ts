/**
 * Named compression-profile presets, bundling the grid/MLP shape knobs
 * (`NTCGridModel.js`'s `GRID_LEVELS_OPTIONS`/`GRID_BASE_RESOLUTION_OPTIONS`/
 * `MLP_HIDDEN_SIZE_OPTIONS`/`MLP_ACTIVATION_OPTIONS`) that would otherwise
 * have to be hand-tuned as 4 independent options, loosely modeled on the
 * NVIDIA neural texture compression paper's own named profiles (Table 2:
 * NTC 0.2/0.5/1.0/2.25 bits-per-pixel-per-channel), but chosen here for
 * decode-cost/compatibility tiers rather than a fixed bpp target - actual
 * storage size and quality both still depend on the source material's
 * resolution and channel count (see NTCModelSize.js's
 * computeModelFootprint for the exact byte accounting).
 *
 * `mobile-fast` mirrors the mobile-oriented defaults sketched in this repo's
 * `EXT_neural_textures.md` draft (1 hidden layer, width 16, targeting
 * roughly <=800 FLOPs/pixel) - it trades reconstruction quality for the
 * smallest decoder (fewest weights, fewest ALU ops per shaded pixel,
 * smallest fp32-uniform footprint and is the safest default for unknown/
 * low-end mobile hardware.
 *
 * `mobile-balanced` is this addon's own long-standing default shape (2
 * hidden layers of width 32, relu) - already a reasonable middle ground
 * that plenty of mobile GPUs handle comfortably, just given a name here
 * rather than being an implicit, undocumented default.
 *
 * `desktop-quality` is the closest match in this codebase to the NVIDIA
 * paper's own default decoder (Section 4.4: 2 hidden layers of width 64,
 * hardGELU) - the highest reconstruction quality of the three, at the
 * highest per-pixel decode cost (4x the hidden-layer weights of
 * mobile-balanced, plus hardGELU's extra ALU ops per neuron over relu's
 * single max() - see NTCMLPTSL.js's evaluateLinearLayerMat4). Best suited to
 * desktop-class GPUs or contexts where decode cost is not the bottleneck.
 *
 * Every profile only sets `levels`/`baseResolution`/`hiddenSizes`/
 * `hiddenActivation` - `channels`, `outputChannels`, `mipsPerLevel`,
 * `quantization`, batch/iteration/learning-rate settings, etc. are left to
 * the caller (NTCTrainer.js's own defaults, or the source material's actual
 * channel layout) exactly as they would be without a profile applied.
 */

interface NTCProfile {
	label: string;
	description: string;
	levels: number;
	baseResolution: number;
	hiddenSizes: number[];
	hiddenActivation: string;
}

const NTC_PROFILES: Record<string, NTCProfile> = {
	'mobile-fast': {
		label: 'Mobile (fast)',
		description: 'Smallest/cheapest decoder - 1 hidden layer x16, relu. Best default for unknown or low-end mobile GPUs.',
		levels: 3,
		baseResolution: 128,
		hiddenSizes: [ 16 ],
		hiddenActivation: 'relu'
	},
	'mobile-balanced': {
		label: 'Mobile (balanced)',
		description: 'This addon\'s long-standing default shape - 2 hidden layers x32, relu.',
		levels: 4,
		baseResolution: 256,
		hiddenSizes: [ 32, 32 ],
		hiddenActivation: 'relu'
	},
	'desktop-quality': {
		label: 'Desktop (quality)',
		description: 'Closest match to the NVIDIA neural texture compression paper\'s own decoder - 2 hidden layers x64, hgelu. Highest quality, highest decode cost.',
		levels: 4,
		baseResolution: 512,
		hiddenSizes: [ 64, 64 ],
		hiddenActivation: 'hgelu'
	}
};

const NTC_PROFILE_NAMES = Object.keys( NTC_PROFILES );

/**
 * Returns the `{ levels, baseResolution, hiddenSizes, hiddenActivation }`
 * shape options for a named profile (see NTC_PROFILES above), or `null` for
 * an unrecognized name - callers merge this into their own options object
 * (e.g. `{ ...options, ...getNTCProfile(name) }`) rather than this module
 * constructing a full NTCTrainer options object itself, since a profile only
 * ever covers the grid/MLP shape, never channel counts, quantization,
 * or training hyperparameters.
 */
function getNTCProfile( name: string ): Pick<NTCProfile, 'levels' | 'baseResolution' | 'hiddenSizes' | 'hiddenActivation'> | null {

	const profile = NTC_PROFILES[ name ];

	if ( ! profile ) return null;

	const { levels, baseResolution, hiddenSizes, hiddenActivation } = profile;

	return { levels, baseResolution, hiddenSizes: hiddenSizes.slice(), hiddenActivation };

}

export {
	NTC_PROFILES,
	NTC_PROFILE_NAMES,
	getNTCProfile
};
export type { NTCProfile };

/** UI controls and training shape derive from the same profile. */
export function getNTCProfileControls(name: string) {
	const profile = getNTCProfile(name);
	if (!profile) return null;
	return {levels:profile.levels, baseResolution:profile.baseResolution,
		hiddenSize:profile.hiddenSizes[0], hiddenLayers:profile.hiddenSizes.length,
		hiddenActivation:profile.hiddenActivation};
}

/** Paper architecture and full training budget. This is intentionally opt-in:
 * 250k batches of 524288 samples are much more expensive than browser defaults.
 * Both grids use uint4 here; this is not a claim of a particular table-2 bitrate.
 */
export function getNTCPaperProfile(textureResolution: number) {
	if (!Number.isInteger(textureResolution) || textureResolution < 16 ||
		!Number.isInteger(Math.log2(textureResolution))) throw new Error('Paper profile requires a power-of-two texture resolution >= 16.');
	const baseResolution = textureResolution / 4;
	return {textureResolution, baseResolution, levels:Math.max(1,Math.floor(Math.log(baseResolution)/Math.log(4))),
		gridChannels:8, lowResChannels:12, mipsPerLevel:2, positionalEncoding:true, positionalEncodingPeriod:8, dualGrid:true,
		hiddenSizes:[64,64], hiddenActivation:'hgelu', batchSize:8*256*256,
		iterations:250000, learningRate:0.01, weightsLearningRate:0.005, cosineAnnealingScale:0,
		quantization:{mode:'uint4'}, retrainAfterQuantize:0.05};
}
