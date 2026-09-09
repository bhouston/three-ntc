// Small trainer-loop utilities shared by every neural trainer (texture,
// material, appearance): a seeded PRNG for reproducible sample generation,
// cosine learning-rate annealing, and yielding a training loop back to the
// browser so a long-running compute loop doesn't freeze the tab.

/**
 * Cosine learning-rate annealing, decaying from `options.learningRate` down
 * toward (not quite) zero over `options.iterations`, floored at
 * `options.cosineAnnealingScale` of the initial rate.
 */
function getLearningRate( options: { learningRate: number; iterations: number; cosineAnnealingScale: number }, iteration: number ): number {

	const t = Math.min( iteration / Math.max( 1, options.iterations - 1 ), 1 );
	const cosine = 0.5 * ( 1 + Math.cos( Math.PI * t ) );
	const scale = options.cosineAnnealingScale + cosine * ( 1 - options.cosineAnnealingScale );

	return options.learningRate * scale;

}

/**
 * A seeded pseudo-random number generator (mulberry32), used instead of
 * `Math.random()` so a training run is reproducible from its seed alone.
 */
function createRandom( seed: number ): () => number {

	let state = seed >>> 0;

	return function random() {

		state = ( state + 0x6D2B79F5 ) | 0;
		let value = Math.imul( state ^ state >>> 15, 1 | state );
		value ^= value + Math.imul( value ^ value >>> 7, 61 | value );

		return ( ( value ^ value >>> 14 ) >>> 0 ) / 4294967296;

	};

}

/**
 * Yields a training loop back to the browser between iterations so the tab
 * stays responsive during a long-running compute loop.
 */
function yieldToBrowser(): Promise<void> {

	return new Promise( ( resolve ) => {

		if ( typeof requestAnimationFrame === 'function' ) {

			requestAnimationFrame( () => resolve() );

		} else {

			setTimeout( resolve, 0 );

		}

	} );

}

export { getLearningRate, createRandom, yieldToBrowser };
