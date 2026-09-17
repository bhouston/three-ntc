import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
const args=process.argv.slice(2);
const arg=(name,fallback)=>{const i=args.indexOf('--'+name);return i<0?fallback:args[i+1];};
const channel=arg('channel','chromium');
const expectedVendor=arg('expected-vendor',process.platform==='win32'?'nvidia':'');
const output=resolve(arg('output','output/benchmarks/gpu-diagnostics.json'));
const report={platform:process.platform,node:process.version,channel,expectedVendor,attempts:[],selected:null};
const server=createServer((_req,res)=>{res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><title>NTC WebGPU diagnostic</title>Checking WebGPU availability…');});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const url=`http://127.0.0.1:${server.address().port}`;
try {
  for(const headed of (args.includes('--headed')?[true]:[false,true])) {
    const attempt={headed,channel};let browser;
    try {
      const launchArgs=['--enable-gpu','--enable-unsafe-webgpu','--ignore-gpu-blocklist',
        ...(process.env.NTC_GPU_BACKEND==='metal'?['--use-angle=metal']:[])];
      browser=await chromium.launch({channel,headless:!headed,args:launchArgs,timeout:30000});
      attempt.browserVersion=browser.version();
      const cdp=await browser.newBrowserCDPSession();
      try {attempt.systemInfo=await cdp.send('SystemInfo.getInfo');}catch(e){attempt.systemInfoError=e.message;}
      const page=await browser.newPage();await page.goto(url);
      attempt.webgpu=await page.evaluate(async()=>{
        const result={secureContext:isSecureContext,gpuApi:!!navigator.gpu,userAgent:navigator.userAgent};
        if(!navigator.gpu)return result;
        try {
          // Match Three.js's default adapter request, rather than validating a
          // different power-preference adapter from the one training will use.
          const adapter=await navigator.gpu.requestAdapter();
          result.adapterAvailable=!!adapter;if(!adapter)return result;
          const info=adapter.info;
          result.adapter=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info?.[k]??'unknown']));
          result.fallback=adapter.isFallbackAdapter ?? info?.isFallbackAdapter ?? null;
          const device=await adapter.requestDevice();
          const module=device.createShaderModule({code:'@compute @workgroup_size(1) fn main() {}'});
          await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
          result.deviceAndComputeAvailable=true;device.destroy();
        }catch(error){result.error=error.message;}
        return result;
      });
      const identity=JSON.stringify(attempt.webgpu.adapter??{}).toLowerCase();
      const vendorMatches=!expectedVendor||identity.includes(expectedVendor.toLowerCase())||(expectedVendor==='nvidia'&&identity.includes('10de'));
      if(attempt.webgpu.deviceAndComputeAvailable&&vendorMatches&&attempt.webgpu.fallback!==true)report.selected={channel,headed,adapter:attempt.webgpu.adapter};
    }catch(error){attempt.error=error.message;}
    finally {if(browser)await browser.close();}
    report.attempts.push(attempt);
    console.log(JSON.stringify({headed,channel,webgpu:attempt.webgpu,error:attempt.error}));
    if(report.selected)break;
  }
}finally{
  server.close();await mkdir(dirname(output),{recursive:true});await writeFile(output,JSON.stringify(report,null,2));
}
console.log('GPU diagnostics: '+output);
if(!report.selected){console.error('No usable matching WebGPU device. Share the diagnostic JSON; training was not started.');process.exitCode=1;}
else console.log(`Selected ${report.selected.channel}, ${report.selected.headed?'visible browser':'headless browser'}.`);
