export function psnr(mse) { return mse===0 ? null : (mse===1 ? 0 : -10*Math.log10(mse)); }
// All classified-active components are evaluated, excluding padding.
// Numerically constant outputs retained by the classifier still count here.
// Fixed physical ranges (2 for signed offsets, 1 otherwise) permit comparisons
// across variants. HDR is reported as unit-reference PSNR, not bounded [0,1] PSNR.
export function measure(reference, decoded, size, componentRanges) {
  const channels=componentRanges.length, pixels=size*size;
  if(reference.length!==pixels*channels || decoded.length!==reference.length) throw new Error('Metric shape mismatch');
  const perComponent=componentRanges.map((range,c)=>{
    let se=0,ae=0,maxError=0,lo=Infinity,hi=-Infinity,ssim=0,blocks=0;
    for(let p=0;p<pixels;p++) {
      const a=reference[p*channels+c],b=decoded[p*channels+c];
      if(!Number.isFinite(a)||!Number.isFinite(b)) throw new Error('Non-finite material output');
      const error=Math.abs(a-b);se+=error*error;ae+=error;maxError=Math.max(maxError,error);lo=Math.min(lo,a);hi=Math.max(hi,a);
    }
    // Explicitly named block SSIM: population statistics in disjoint 8x8 windows.
    // This is not the Gaussian-window SSIM used by some image metric packages.
    for(let y=0;y<size;y+=8) for(let x=0;x<size;x+=8) {
      let a=0,b=0,aa=0,bb=0,ab=0,n=0;
      for(let dy=0;dy<Math.min(8,size-y);dy++) for(let dx=0;dx<Math.min(8,size-x);dx++) {
        const i=((y+dy)*size+x+dx)*channels+c,r=reference[i],d=decoded[i];
        a+=r;b+=d;aa+=r*r;bb+=d*d;ab+=r*d;n++;
      }
      a/=n;b/=n;const c1=(0.01*range)**2,c2=(0.03*range)**2;
      ssim+=((2*a*b+c1)*(2*(ab/n-a*b)+c2))/((a*a+b*b+c1)*(Math.max(0,aa/n-a*a)+Math.max(0,bb/n-b*b)+c2));blocks++;
    }
    return {mse:se/pixels,normalizedMse:se/pixels/range**2,mae:ae/pixels,maxError,referenceMin:lo,referenceMax:hi,blockSsim:ssim/blocks};
  });
  const normalizedMse=perComponent.reduce((s,c)=>s+c.normalizedMse,0)/channels;
  return {size,pixels,normalizedMse,psnr:psnr(normalizedMse),blockSsim:perComponent.reduce((s,c)=>s+c.blockSsim,0)/channels,perComponent};
}
export function summarizeMips(mips) {
  const pixels=mips.reduce((s,m)=>s+m.pixels,0);
  const mse=mips.reduce((s,m)=>s+m.normalizedMse*m.pixels,0)/pixels;
  const equalMipMse=mips.reduce((s,m)=>s+m.normalizedMse,0)/mips.length;
  return {mip0Psnr:mips[0].psnr,normalizedMse:mse,psnr:psnr(mse),equalMipMse,equalMipPsnr:psnr(equalMipMse),mips};
}
export function pairedStats(values) {
  if(!values.length)return null;
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  const sd=values.length>1?Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/(values.length-1)):null;
  // Student-t 95% intervals for seed replicates; do not treat texels as replicates.
  const critical={2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}[values.length];
  const halfWidth=critical?critical*sd/Math.sqrt(values.length):null;
  return {n:values.length,mean,sd,low:halfWidth===null?null:mean-halfWidth,high:halfWidth===null?null:mean+halfWidth};
}

/** Angular error after the library's positive-hemisphere normal reconstruction. */
export function normalError(reference, decoded, channels, offset) {
  let sum=0,sumSquares=0,maximum=0,invalidDisk=0;const count=reference.length/channels;
  for(let p=0;p<count;p++) {
    const i=p*channels+offset,ax=reference[i],ay=reference[i+1],bx=decoded[i],by=decoded[i+1];
    const az=Math.sqrt(Math.max(0,1-ax*ax-ay*ay)),bz=Math.sqrt(Math.max(0,1-bx*bx-by*by));
    const dot=(ax*bx+ay*by+az*bz)/(Math.hypot(ax,ay,az)*Math.hypot(bx,by,bz));
    const degrees=Math.acos(Math.min(1,Math.max(-1,dot)))*180/Math.PI;
    if(!Number.isFinite(degrees))throw new Error('Non-finite reconstructed normal');
    sum+=degrees;sumSquares+=degrees*degrees;maximum=Math.max(maximum,degrees);
    if(bx*bx+by*by>1)invalidDisk++;
  }
  return {meanDegrees:sum/count,rmsDegrees:Math.sqrt(sumSquares/count),maxDegrees:maximum,outsideUnitDiskFraction:invalidDisk/count};
}
