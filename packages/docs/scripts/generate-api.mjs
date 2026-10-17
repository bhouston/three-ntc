import { Application } from 'typedoc';
import { mkdir, rm, writeFile } from 'node:fs/promises';

// Generate each published entry point independently, retaining its public export surface.
for (const [name, packageName] of [
  ['runtime', 'three-ntc'],
  ['trainer', 'three-ntc-trainer'],
]) {
  const out = `docs/api/${name}`;
  await rm(out, { recursive: true, force: true });
  const app = await Application.bootstrapWithPlugins({
    entryPoints: [`../${packageName}/src/index.ts`],
    tsconfig: `../${packageName}/tsconfig.json`,
    plugin: ['typedoc-plugin-markdown'],
    out,
    outputs: [{ name: 'markdown', path: out }],
    name: packageName,
    readme: 'none',
    entryFileName: 'index',
    hidePageHeader: true,
    hideBreadcrumbs: true,
    disableSources: true,
    skipErrorChecking: true,
    // Workspace type checking uses the pinned compiler separately; TypeDoc uses TS 5.9.
    compilerOptions: { ignoreDeprecations: '5.0' },
  });
  const project = await app.convert();
  if (!project) throw new Error(`Could not generate ${packageName} API`);
  await app.generateOutputs(project);
  await mkdir(out, { recursive: true });
  await writeFile(
    `${out}/_category_.json`,
    JSON.stringify({ label: `${packageName} API`, position: name === 'runtime' ? 10 : 11 }),
  );
}
