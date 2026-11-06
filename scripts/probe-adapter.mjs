import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
const base = ['--enable-unsafe-webgpu','--enable-features=Vulkan','--ignore-gpu-blocklist','--disable-vulkan-surface','--disable-gpu-driver-bug-workarounds','--disable-gpu-watchdog','--no-sandbox'];
const V = {
  H_raw: [],
  Q_swangle: ['--use-angle=vulkan','--use-gl=angle','--use-vulkan=native','--disable-gpu-sandbox','--in-process-gpu'],
  R_inproc: ['--in-process-gpu'],
  J_native: ['--use-vulkan=native'],
  K_angle_vk: ['--use-gl=angle','--use-angle=vulkan'],
  L_nosw: ['--disable-software-rasterizer'],
  M_nogpusandbox: ['--disable-gpu-sandbox'],
  N_adapter_vulkan: ['--use-webgpu-adapter=vulkan'],
  O_dawn_feat: ['--disable-dawn-features=disallow_unsafe_apis','--enable-dawn-features=use_vulkan_loader_icd'],
  P_noenable_feat: ['--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan'],
};
for (const [name, extra] of Object.entries(V)) {
  try {
    const b = await chromium.launch({ headless: false, channel: 'chrome', ignoreDefaultArgs: ['--enable-unsafe-swiftshader'], args: [...base, ...extra], env: { ...process.env, VK_DRIVER_FILES: '/usr/share/vulkan/icd.d/lvp_icd.json', VK_ICD_FILENAMES: '/usr/share/vulkan/icd.d/lvp_icd.json' } });
    const p = await b.newPage();
    await p.route('http://localhost:1/', r => r.fulfill({ body: '<html></html>', contentType: 'text/html' }));
    await p.goto('http://localhost:1/');
    const r = await p.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? JSON.stringify([a.info.vendor, a.info.architecture, a.info.description]) : 'none'; });
    console.log('PROBE', name, r);
    await b.close();
  } catch (e) { console.log('PROBE', name, 'ERR', String(e).slice(0, 200)); }
}
