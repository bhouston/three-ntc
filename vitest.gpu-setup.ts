// Log the WebGPU adapter so CI output shows whether a real/lavapipe/SwiftShader device ran.
const adapter = await navigator.gpu?.requestAdapter();
console.log('NTC_ADAPTER ' + JSON.stringify(adapter?.info ?? null));
