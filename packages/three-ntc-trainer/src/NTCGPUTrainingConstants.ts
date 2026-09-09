// Fixed-point scale factors shared by every neural training system
// (texture, material, appearance) that scatters gradients into WebGPU
// storage buffers via `atomicAdd`. WGSL has no float atomics, so every one
// of these systems quantizes gradients to integers before accumulating and
// divides back down after. Kept in one place so the scale never drifts
// between systems that otherwise train completely different models.

const FIXED_POINT_SCALE = 1e5;
const GRADIENT_NORM_SCALE = 1e5;

export { FIXED_POINT_SCALE, GRADIENT_NORM_SCALE };
