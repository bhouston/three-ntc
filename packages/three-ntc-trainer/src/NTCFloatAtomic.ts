import { wgslFn, bitcast, atomicLoad } from 'three/tsl';

// Store IEEE-754 bits in an integer atomic, avoiding fixed-point overflow and
// per-sample truncation. Retry weak CAS until this contribution is committed.
export const FLOAT_ATOMIC_WGSL = `
fn ntcAtomicAddFloat(destination: ptr<storage, atomic<i32>, read_write>, value: f32) -> f32 {
  if (value == 0.0) { return 0.0; }
  var previous = atomicLoad(destination);
  loop {
    let updated = bitcast<i32>(bitcast<f32>(previous) + value);
    let result = atomicCompareExchangeWeak(destination, previous, updated);
    if (result.exchanged) { break; }
    previous = result.old_value;
  }
  return bitcast<f32>(previous);
}
`;
const addFloat = wgslFn(FLOAT_ATOMIC_WGSL);
export function atomicAddFloat(destination: any, value: any): void {
  addFloat(destination, value).toVar();
}
export function atomicLoadFloat(destination: any): any {
  return bitcast(atomicLoad(destination), 'float');
}
