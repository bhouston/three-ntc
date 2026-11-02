// Log the WebGPU adapter so CI output shows whether a real/lavapipe/SwiftShader device ran.
const adapter = await navigator.gpu?.requestAdapter();
const { vendor, architecture, device, description } = adapter?.info ?? {};
console.log('NTC_ADAPTER ' + JSON.stringify({ vendor, architecture, device, description }));
