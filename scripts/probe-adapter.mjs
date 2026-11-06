import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
const base = ['--enable-unsafe-webgpu','--enable-features=Vulkan','--ignore-gpu-blocklist','--disable-vulkan-surface','--disable-gpu-driver-bug-workarounds','--disable-gpu-watchdog','--no-sandbox'];
const V = {
  H_raw: [],
  J_native: ['--use-vulkan=native'],
  K_angle_vk: ['--use-gl=angle','--use-angle=vulkan'],
  L_nosw: ['--disable-software-rasterizer'],
  M_nogpusandbox: ['--disable-gpu-sandbox'],
  N_adapter_vulkan: ['--use-webgpu-adapter=vulkan'],
  O_dawn_feat: ['--disable-dawn-features=disallow_unsafe_apis','--enable-dawn-features=use_vulkan_loader_icd'],
  P_noenable_feat: ['--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan'],
};
let port = 9400;
for (const [name, extra] of Object.entries(V)) {
  const pt = port++;
  const c = spawn('google-chrome', [...base, ...extra, '--enable-logging=stderr', '--v=0', `--remote-debugging-port=${pt}`, '--user-data-dir=/tmp/probe-' + name, 'about:blank'], { env: { ...process.env, VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/lvp_icd.json', VK_ICD_FILENAMES: '/usr/share/vulkan/icd.d/lvp_icd.json' }, stdio: ['ignore', 'ignore', 'pipe'] });
  let err = ''; c.stderr.on('data', d => err += d);
  await new Promise(r => setTimeout(r, 4000));
  try {
    const b = await chromium.connectOverCDP(`http://127.0.0.1:${pt}`);
    const p = await b.contexts()[0].newPage();
    await p.route('http://localhost:1/', r => r.fulfill({ body: '<html></html>', contentType: 'text/html' }));
    await p.goto('http://localhost:1/');
    const r = await p.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? JSON.stringify([a.info.vendor, a.info.architecture, a.info.description]) : 'none'; });
    console.log('PROBE', name, r);
    await b.close();
  } catch (e) { console.log('PROBE', name, 'ERR', String(e).slice(0, 200)); }
  c.kill();
  console.log('STDERR', name, err.split('\n').filter(l => /vulkan|swiftshader|dawn|lvp|llvmpipe|gpu/i.test(l)).slice(0, 8).join(' | ').slice(0, 1200));
}
