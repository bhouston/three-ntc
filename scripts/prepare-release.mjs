import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

export const packages = ['three-ntc', 'three-ntc-trainer'];

// Only the CI checkout is modified. Git tags are the version source of truth.
//
// This module runs as two separate "prepare" plugin entries in
// release.config.js, wrapped around @anolilab/semantic-release-pnpm's own
// two `prepare` entries (one per package):
//
//   1. files (default): stage LICENSE/CHANGELOG.md into each package dir and
//      register CHANGELOG.md in `files`, before pnpm bumps versions and
//      packs/publishes.
//   2. artifacts (`{ artifacts: true }`, run last): once both packages have
//      been version-bumped, pack each into release-artifacts/ for the GitHub
//      release. This is NOT delegated to @anolilab/semantic-release-pnpm's
//      own `tarballDir` option, because (verified against pnpm 11.1.3 and
//      11.26.0) `pnpm pack <dir>` ignores a directory argument and always
//      packs the current working directory's package, unlike `pnpm publish
//      <dir>` (which the plugin's actual registry-publish step uses and
//      which does respect it). `pnpm --dir <pkgRoot> pack` is the correct,
//      pnpm-native invocation, and it also natively resolves the trainer's
//      `workspace:^` dependency on three-ntc to a real semver range.
export function prepare({ artifacts } = {}, { cwd, env }) {
  if (artifacts) {
    const packDestination = join(cwd, 'release-artifacts');
    for (const name of packages) {
      execFileSync('pnpm', ['--dir', join(cwd, 'packages', name), 'pack', '--pack-destination', packDestination], {
        cwd,
        env,
        stdio: 'pipe',
      });
    }
    return;
  }
  for (const name of packages) {
    const dir = join(cwd, 'packages', name);
    const path = join(dir, 'package.json');
    const pkg = JSON.parse(readFileSync(path, 'utf8'));
    pkg.files = [...new Set([...pkg.files, 'CHANGELOG.md'])];
    writeFileSync(path, `${JSON.stringify(pkg, null, 2)}\n`);
    copyFileSync(join(cwd, 'LICENSE'), join(dir, 'LICENSE'));
    copyFileSync(join(cwd, 'CHANGELOG.md'), join(dir, 'CHANGELOG.md'));
  }
}
