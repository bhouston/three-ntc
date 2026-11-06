import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { analyzeCommits } from '@semantic-release/commit-analyzer';

// Published via pnpm publish directly (see release.config.js): pnpm rewrites
// workspace:^ deps and packs the `files` field natively, so these packages
// just need a real LICENSE and a `files` field that ships `dist`.
const packages = ['three-ntc', 'three-ntc-trainer'].map((name) => `packages/${name}`);

for (const path of packages) {
  void test(`${path} is ready for pnpm publish`, () => {
    assert.ok(existsSync(resolve(path, 'LICENSE')), `${path}/LICENSE is missing`);
    const pkg = JSON.parse(readFileSync(resolve(path, 'package.json'), 'utf8'));
    assert.ok(pkg.files?.includes('dist'), `${path}/package.json files must include "dist"`);
  });
}

for (const [message, expected] of [
  ['fix: handle empty textures', 'patch'],
  ['feat: add exporter', 'minor'],
  ['feat!: replace the public loader API', 'major'],
  ['fix: change loader\n\nBREAKING CHANGE: remove the legacy API', 'major'],
  ['docs: explain installation', null],
  ['chore: update CI', null],
]) {
  void test(`release level for ${message.split('\n')[0]}`, async () => {
    assert.equal(
      await analyzeCommits(
        { preset: 'conventionalcommits' },
        {
          cwd: process.cwd(),
          commits: [{ hash: 'abc123', message }],
          logger: { log() {} },
        },
      ),
      expected,
    );
  });
}

void test('the real release pipeline (per-package version bump, inline artifact pack) produces installable registry tarballs, with pnpm natively resolving workspace:^ to a real semver range', async () => {
  const { cpSync } = await import('node:fs');
  const { execFileSync } = await import('node:child_process');
  const { prepare: pnpmPrepare } = await import('@anolilab/semantic-release-pnpm');
  const { default: config } = await import('../release.config.js');
  const repo = process.cwd();
  const cwd = mkdtempSync(join(tmpdir(), 'ntc-pack-'));
  try {
    // A real (if minimal) pnpm workspace, so `pnpm pack` can resolve the
    // trainer's `workspace:^` dependency on three-ntc the same way it does
    // in CI, instead of mocking that resolution.
    writeFileSync(join(cwd, 'pnpm-workspace.yaml'), 'packages:\n  - packages/*\n');
    writeFileSync(
      join(cwd, 'package.json'),
      JSON.stringify({ name: 'ntc-pack-test-root', version: '0.0.0', private: true }),
    );
    for (const path of packages) {
      const dir = join(cwd, path);
      mkdirSync(dir, { recursive: true });
      cpSync(join(repo, path, 'dist'), join(dir, 'dist'), { recursive: true });
      for (const file of ['package.json', 'README.md', 'LICENSE']) cpSync(join(repo, path, file), join(dir, file));
    }
    const context = {
      cwd,
      env: process.env,
      nextRelease: { version: '0.2.0' },
      stdout: process.stdout,
      stderr: process.stderr,
      logger: { log() {} },
    };
    execFileSync('pnpm', ['install', '--no-frozen-lockfile', '--ignore-scripts'], { cwd, stdio: 'pipe' });
    // Mirrors release.config.js's plugin order: each package's pnpm version
    // bump, then the config's own inline prepare (artifact pack, relative to cwd).
    for (const path of packages) await pnpmPrepare({ pkgRoot: path, npmPublish: false }, context);
    const inline = config.plugins.find((plugin) => typeof plugin === 'object' && !Array.isArray(plugin));
    process.chdir(cwd);
    try {
      inline.prepare();
    } finally {
      process.chdir(repo);
    }
    for (const path of packages) {
      const name = path.replace('packages/', '');
      const tarball = join(cwd, 'release-artifacts', `${name}-0.2.0.tgz`);
      assert.ok(existsSync(tarball), `${name}: tarball not found at ${tarball}`);
      const entries = execFileSync('tar', ['-tf', tarball], { encoding: 'utf8' });
      for (const file of ['dist/index.js', 'dist/index.d.ts', 'LICENSE', 'README.md']) {
        assert.ok(entries.includes(`package/${file}`), `${name}: missing ${file}`);
      }
      assert.ok(!entries.includes('CHANGELOG.md'), `${name}: CHANGELOG.md must not be shipped`);
      const pkg = JSON.parse(execFileSync('tar', ['-xOf', tarball, 'package/package.json'], { encoding: 'utf8' }));
      assert.equal(pkg.version, '0.2.0');
      assert.ok(!JSON.stringify(pkg).includes('workspace:'));
      if (name.endsWith('trainer')) assert.equal(pkg.dependencies['three-ntc'], '^0.2.0');
    }
  } finally {
    process.chdir(repo);
    rmSync(cwd, { recursive: true, force: true });
  }
});

void test('renders release notes with the installed Conventional Commits preset', async () => {
  const { generateNotes } = await import('@semantic-release/release-notes-generator');
  const notes = await generateNotes(
    { preset: 'conventionalcommits' },
    {
      cwd: process.cwd(),
      options: { repositoryUrl: 'https://github.com/bhouston/three-ntc.git' },
      lastRelease: { gitTag: 'v0.1.0', version: '0.1.0' },
      nextRelease: { gitTag: 'v0.2.0', version: '0.2.0' },
      commits: [
        { hash: '1234567890abcdef', message: 'feat: add texture export' },
        { hash: 'abcdef1234567890', message: 'fix: preserve mip levels' },
        { hash: 'fedcba1234567890', message: 'feat!: replace loader options' },
      ],
      logger: { log() {} },
    },
  );
  assert.match(notes, /add texture export/);
  assert.match(notes, /preserve mip levels/);
  assert.match(notes, /BREAKING CHANGES/);
  assert.match(notes, /v0\.1\.0\.\.\.v0\.2\.0/);
});
