/**
 * Draws one or more named, colored training-loss curves (log-scale y axis,
 * training-iteration x axis) onto a <canvas>, and keeps an optional legend
 * element in sync with the same labels/colors. Shared by the neural-appearance,
 * neural-material and neural-texture examples, which otherwise each carried a
 * near-identical hand-rolled version of this - some tracking a single "loss"
 * series, others (neural-appearance) several at once (training/direct/ibl/
 * validation loss).
 */

interface NTCLossSeries {
	key: string;
	label: string;
	color: string;
}

type NTCLossPoint = Record<string, number> & { iteration: number };

class NTCLossGraph {

	canvas: HTMLCanvasElement;
	context: CanvasRenderingContext2D;
	series: NTCLossSeries[];
	legendElement: HTMLElement | null;
	ipsElement: HTMLElement | null;
	history: NTCLossPoint[];
	iterationBasis: number | null;
	iterationSamples: Array<{ time: number; iteration: number }>;

	// `series` is an array of `{ key, label, color }`: `key` selects the value
	// out of each history point, `label`/`color` drive both the drawn line and,
	// if `legend` is given, an auto-built legend entry.
	constructor( canvas: HTMLCanvasElement, series: NTCLossSeries[], { legend = null, ips = null }: { legend?: HTMLElement | null; ips?: HTMLElement | null } = {} ) {

		this.canvas = canvas;
		this.context = canvas.getContext( '2d' ) as CanvasRenderingContext2D;
		this.series = series;
		this.legendElement = legend;
		this.ipsElement = ips;
		this.history = [];

		// When left null, the x axis scales against the last recorded point's
		// iteration (matches the single-series examples' prior behavior).
		// setIterationBasis() lets a caller pin it to the configured training
		// length instead, so the axis doesn't keep rescaling as points arrive.
		this.iterationBasis = null;

		// Rolling window of { time, iteration } samples covering roughly the
		// last second, used to derive an iterations/second readout without
		// needing the caller to track timing itself.
		this.iterationSamples = [];

		if ( this.legendElement ) this.buildLegend();

	}

	buildLegend() {

		this.legendElement!.textContent = '';

		for ( const { label, color } of this.series ) {

			const span = document.createElement( 'span' );
			span.textContent = label;
			span.style.color = color;
			this.legendElement!.appendChild( span );

		}

	}

	setIterationBasis( iterations: number ) {

		this.iterationBasis = iterations;

	}

	reset() {

		this.history = [];
		this.iterationSamples = [];
		if ( this.ipsElement ) this.ipsElement.textContent = '';
		this.draw();

	}

	addPoint( point: NTCLossPoint ) {

		this.history.push( point );

		const now = performance.now();
		this.iterationSamples.push( { time: now, iteration: point.iteration } );

		// Drop samples older than one second, but always keep at least one
		// sample past the window edge so the elapsed time used below spans
		// (approximately) a full second rather than whatever's left inside it.
		while ( this.iterationSamples.length > 2 && now - this.iterationSamples[ 1 ].time > 1000 ) {

			this.iterationSamples.shift();

		}

		if ( this.ipsElement ) {

			const ips = this.getIterationsPerSecond();
			this.ipsElement.textContent = ips !== null ? `${ ips.toFixed( 1 ) } it/s` : '';

		}

	}

	// Iterations per second, averaged over the current rolling window
	// (spanning up to the last second of addPoint() calls). Returns null
	// until at least two samples have been recorded.
	getIterationsPerSecond(): number | null {

		if ( this.iterationSamples.length < 2 ) return null;

		const first = this.iterationSamples[ 0 ];
		const last = this.iterationSamples[ this.iterationSamples.length - 1 ];
		const elapsedSeconds = ( last.time - first.time ) / 1000;

		if ( elapsedSeconds <= 0 ) return null;

		return ( last.iteration - first.iteration ) / elapsedSeconds;

	}

	draw() {

		const context = this.context;
		const width = this.canvas.width;
		const height = this.canvas.height;
		const marginLeft = 42;
		const marginRight = 10;
		const marginTop = 12;
		const marginBottom = 24;
		const plotWidth = width - marginLeft - marginRight;
		const plotHeight = height - marginTop - marginBottom;

		context.clearRect( 0, 0, width, height );
		context.fillStyle = '#050505';
		context.fillRect( 0, 0, width, height );
		drawAxes( context, marginLeft, marginTop, plotWidth, plotHeight );

		const lastPoint = this.history.length > 0 ? this.history[ this.history.length - 1 ] : null;
		const lastIteration = lastPoint ? lastPoint.iteration : null;
		const primarySeries = this.series[ 0 ];
		const primaryValue = lastPoint && primarySeries ? lastPoint[ primarySeries.key ] : undefined;

		context.fillStyle = '#aaa';
		context.font = '10px monospace';
		context.fillText( 'loss (log)', marginLeft, 9 );

		// The primary (first) series' most recent value, read straight out of
		// history rather than passed in separately - it's always just the
		// last point's value for series[ 0 ].
		context.textAlign = 'right';
		context.fillStyle = primarySeries ? primarySeries.color : '#aaa';
		context.fillText( Number.isFinite( primaryValue ) ? ( primaryValue as number ).toExponential( 3 ) : '—', width - marginRight, 9 );

		// Doubles as the x axis's extent indicator - the axis always scales to
		// exactly this iteration (see maxIteration below), so showing the
		// number here instead of a static "iteration" label is what tells the
		// reader what the right edge of the plot currently means.
		context.fillStyle = '#aaa';
		context.fillText( lastIteration !== null ? `iteration ${lastIteration}` : 'iteration', width - marginRight, height - 4 );
		context.textAlign = 'left';

		const values = this.history
			.flatMap( ( point ) => this.series.map( ( entry ) => point[ entry.key ] ) )
			.filter( ( value ) => Number.isFinite( value ) && value > 0 );

		if ( values.length === 0 ) {

			context.fillText( 'Train to graph loss.', marginLeft + 30, marginTop + plotHeight / 2 );
			return;

		}

		let minLog = Math.log10( Math.min( ...values ) );
		let maxLog = Math.log10( Math.max( ...values ) );

		if ( minLog === maxLog ) {

			minLog -= 0.5;
			maxLog += 0.5;

		}

		drawScaleLabels( context, marginLeft, marginTop, plotHeight, minLog, maxLog );

		const maxIteration = Math.max( 1, ( this.iterationBasis !== null ? this.iterationBasis : lastIteration! ) - 1 );

		for ( const { key, color } of this.series ) {

			drawSeries( context, this.history, key, color, marginLeft, marginTop, plotWidth, plotHeight, minLog, maxLog, maxIteration );

		}

	}

}

function drawAxes( context: CanvasRenderingContext2D, x: number, y: number, width: number, height: number ) {

	context.strokeStyle = 'rgba( 255, 255, 255, 0.35 )';
	context.lineWidth = 1;
	context.beginPath();
	context.moveTo( x, y );
	context.lineTo( x, y + height );
	context.lineTo( x + width, y + height );
	context.stroke();

}

function drawScaleLabels( context: CanvasRenderingContext2D, _x: number, y: number, height: number, minLog: number, maxLog: number ) {

	context.fillStyle = '#777';
	context.font = '10px monospace';
	context.fillText( Math.pow( 10, maxLog ).toExponential( 1 ), 2, y + 4 );
	context.fillText( Math.pow( 10, minLog ).toExponential( 1 ), 2, y + height );

}

function drawSeries( context: CanvasRenderingContext2D, history: NTCLossPoint[], key: string, color: string, x: number, y: number, width: number, height: number, minLog: number, maxLog: number, maxIteration: number ) {

	const points = history.filter( ( point ) => Number.isFinite( point[ key ] ) && point[ key ] > 0 );
	if ( points.length === 0 ) return;

	const logRange = maxLog - minLog || 1;

	context.strokeStyle = color;
	context.lineWidth = 2;
	context.beginPath();

	for ( let i = 0; i < points.length; i ++ ) {

		const point = points[ i ];
		const px = x + ( point.iteration - 1 ) / maxIteration * width;
		const py = y + height - ( Math.log10( point[ key ] ) - minLog ) / logRange * height;

		if ( i === 0 ) context.moveTo( px, py );
		else context.lineTo( px, py );

	}

	context.stroke();

}

export { NTCLossGraph };
export type { NTCLossSeries, NTCLossPoint };
