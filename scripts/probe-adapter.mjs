import { chromium } from 'playwright';
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
for (const [name, o] of Object.entries(variants)) {
  try {
    const b = await chromium.launch({ headless: false, ...o });
    const p = await b.newPage();
    await p.goto('about:blank');
    const r = await p.evaluate(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? JSON.stringify(a.info && { v: a.info.vendor, a: a.info.architecture, d: a.info.description }) : 'none'; });
    console.log('PROBE', name, r, b.version());
    await b.close();
  } catch (e) { console.log('PROBE', name, 'ERR', String(e).slice(0, 200)); }
}
