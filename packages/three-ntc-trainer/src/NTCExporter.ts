import { encodeNTC } from './NTCManifest.js';

/**
 * An exporter for `.ntc` (Neural Texture Compression) assets.
 *
 * Encodes an already-trained `NTCNodeMaterial` (see the sibling `three-ntc`
 * runtime package's `NTCNodeMaterial.js`, and this package's `NTCTrainer.js`)
 * - its shared multiresolution latent grid plus MLP decoder, and the channel
 * layout it was fit against - into the compact JSON manifest `NTCLoader`
 * reads back (see `NTCManifest.js` for the format itself).
 *
 * `material` must be an `NTCNodeMaterial` instance (or any object exposing
 * the same `cpuModel`/`activeChannels`/`channels` fields that constructor
 * leaves on `this` - see NTCNodeMaterial.js), not a general `Material`;
 * there is no path from an arbitrary material back to a trained grid+MLP
 * model.
 *
 * ```js
 * const exporter = new NTCExporter();
 * const manifest = exporter.parse( ntcNodeMaterial, { name: 'Gold' } );
 * const blob = new Blob( [ JSON.stringify( manifest ) ], { type: 'application/json' } );
 * ```
 */
class NTCExporter {

	/**
	 * Parses the given trained NTC material and generates the `.ntc` manifest.
	 *
	 * @param material - A trained `NTCNodeMaterial` (must carry `cpuModel`, `activeChannels`, `channels`).
	 * @param options - The export options.
	 * @return The `.ntc` manifest - JSON-serializable as-is (`JSON.stringify( manifest )`).
	 */
	parse( material: any, options: NTCExporterOptions = {} ): any {

		if ( ! material || ! material.cpuModel || ! material.activeChannels ) {

			throw new Error( 'THREE.NTCExporter: material must be a trained NTCNodeMaterial (missing cpuModel/activeChannels).' );

		}

		const channelClassification = {
			activeChannels: material.activeChannels,
			constantValues: material._constantValues || {},
			totalChannels: material.cpuModel.outputChannels,
			packCount: Math.ceil( material.cpuModel.outputChannels / 4 ),
			renderFlags: { side: material.side, transparent: material.transparent }
		};

		return encodeNTC( material.cpuModel, channelClassification, options );

	}

}

/**
 * NTC exporter options.
 */
interface NTCExporterOptions {
	/** A display name embedded in the manifest. */
	name?: string;
	/** A free-form provenance string embedded in the manifest. */
	source?: string;
	/** The latent grid's wrap mode. */
	wrap?: 'repeat' | 'clamp';
	/** Explicit per-level `[min, max]` quantization ranges, overriding both `material.cpuModel.quantizationRange` (QAT) and a plain min/max scan. */
	quantizationRanges?: Array<[ number, number ]>;
	/** Overrides `material.cpuModel.uvTransform` (see `NTCNodeMaterial.js`) - a mesh/query-UV-to-local-space affine transform. Omitted from the manifest when identity/not supplied. */
	uvTransform?: unknown;
}

export { NTCExporter };
export type { NTCExporterOptions };
