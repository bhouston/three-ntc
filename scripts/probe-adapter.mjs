import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
const base = ['--enable-unsafe-webgpu','--enable-features=Vulkan','--ignore-gpu-blocklist','--disable-vulkan-surface','--disable-gpu-driver-bug-workarounds','--disable-gpu-watchdog','--no-sandbox'];
const variants = {
  A_current: { args: base, channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
  B_cft: { args: base, ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
  C_cft_default: { args: base },
  D_vulkan_native: { args: [...base, '--use-vulkan=native'], channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
  E_adapter_default: { args: [...base, '--use-webgpu-adapter=default'], channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
  F_angle_vk: { args: [...base, '--use-angle=vulkan', '--use-gl=angle'], channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
  G_disable_features: { args: [...base, '--disable-features=VulkanFromANGLE,DefaultANGLEVulkan'], channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] },
};

const probe = async (name, b) => {
  const p = await b.newPage();
  await p.route('http://localhost:1/', r => r.fulfill({ body: '<html></html>', contentType: 'text/html' }));
  await p.goto('http://localhost:1/');
  const r = await p.evaluate(async () => { const o = {}; for (const f of [false, true]) { const a = await navigator.gpu?.requestAdapter({ forceFallbackAdapter: f }); o[f] = a ? [a.info.vendor, a.info.architecture, a.info.description, a.isFallbackAdapter] : null; } return JSON.stringify(o); });
  console.log('PROBE', name, r);
  return p;
};
// H: raw chrome via CDP, like puppeteer
for (const [name, extra] of [['H_raw', []], ['I_raw_headless', ['--headless=new']]]) {
  const c = spawn('google-chrome', [...base, ...extra, '--remote-debugging-port=9333', '--user-data-dir=/tmp/probe-' + name, 'about:blank'], { env: { ...process.env, VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/lvp_icd.json' }, stdio: 'ignore' });
  await new Promise(r => setTimeout(r, 3000));
  try { const b = await chromium.connectOverCDP('http://localhost:9333'); const p = await probe(name, b); await p.goto('chrome://gpu'); await new Promise(r => setTimeout(r, 2000)); const t = await p.evaluate(() => document.body.innerText); console.log('GPUINFO', name, t.split('\n').filter(l => /vulkan|dawn|adapter|swiftshader|lavapipe|llvmpipe|webgpu|GL_RENDERER|Problems/i.test(l)).slice(0, 25).join(' | ')); await b.close(); } catch (e) { console.log('PROBE', name, 'ERR', String(e).slice(0, 200)); }
  c.kill();
}
for (const [name, o] of Object.entries(variants)) {
  try {
    const b = await chromium.launch({ headless: false, ...o });
    
    await probe(name, b);
    await b.close();
  } catch (e) { console.log('PROBE', name, 'ERR', String(e).slice(0, 200)); }
}
