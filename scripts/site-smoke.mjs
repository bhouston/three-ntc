import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

const base = process.env.SITE_TEST_URL ?? 'http://127.0.0.1:3300/three-ntc/';
const artifacts = process.env.SWARM_ARTIFACTS;
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const report = { base, gpu: null, pages: [], training: null, errors: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  page.on('pageerror', (error) => report.errors.push(error.message));
  page.on('response', (response) => {
    if (response.status() >= 400) report.errors.push(`${response.status()} ${response.url()}`);
  });
  await page.goto(base);
  report.gpu = await page.evaluate(async () => {
    const gl = document.createElement('canvas').getContext('webgl2');
    const ext = gl?.getExtension('WEBGL_debug_renderer_info');
    const adapter = await navigator.gpu?.requestAdapter();
    return {
      renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null,
      vendor: adapter?.info.vendor,
      architecture: adapter?.info.architecture,
      fallback: adapter?.info.isFallbackAdapter,
    };
  });
  if (!report.gpu.vendor || /swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(report.gpu)) || report.gpu.fallback) {
    throw new Error(`Hardware GPU unavailable: ${JSON.stringify(report.gpu)}`);
  }
  for (const [name, path] of [
    ['home', ''],
    ['runtime-api', 'docs/api/runtime/classes/NTCLoader/'],
    ['theory', 'docs/theory/'],
    ['demo-page', 'demo/'],
    ['trainer-page', 'trainer/'],
    ['gallery', 'tools/?view=gallery'],
    ['viewer', 'tools/?view=viewer'],
    ['trainer', 'tools/?view=trainer'],
  ]) {
    const response = await page.goto(new URL(path, base).href);
    await page.waitForLoadState('networkidle');
    if (path.startsWith('tools/')) {
      await page.locator('canvas').first().waitFor({ timeout: 30000 });
      await page.waitForTimeout(4000);
    }
    if (name === 'demo-page' || name === 'trainer-page') {
      await page.frameLocator('iframe').locator('canvas').first().waitFor({ timeout: 30000 });
    }
    if (name === 'trainer') {
      // A small real GPU fit exercises baking, compute, progress, and export.
      await page.getByRole('slider').first().focus();
      await page.keyboard.press('Home');
      await page.getByRole('button', { name: 'Train', exact: true }).click();
      const save = page.getByRole('button', { name: 'Save .ntc', exact: true });
      await save.waitFor();
      await page.waitForFunction(
        () =>
          ![...document.querySelectorAll('button')].find((button) => button.textContent.trim() === 'Save .ntc')
            ?.disabled,
        { timeout: 120000 },
      );
      const downloading = page.waitForEvent('download');
      await save.click();
      const download = await downloading;
      if (artifacts) await download.saveAs(`${artifacts}/trained-material.ntc`);
      report.training = { iterations: 200, download: download.suggestedFilename() };
    }
    if (artifacts) {
      await mkdir(artifacts, { recursive: true });
      await page.screenshot({ path: `${artifacts}/${name}.png`, fullPage: true });
    }
    report.pages.push({ name, status: response.status(), canvases: await page.locator('canvas').count() });
  }
  if (report.errors.length) throw new Error(`Browser errors: ${report.errors.join('; ')}`);
} finally {
  await browser.close();
  if (artifacts) await writeFile(`${artifacts}/browser-report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
