import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { prepare, packages } from './prepare-release.mjs';
import { analyzeCommits } from '@semantic-release/commit-analyzer';

test('prepares release documents without touching version or workspace dependencies', () => {
  const cwd = mkdtempSync(join(tmpdir(), 'ntc-release-'));
  try {
    writeFileSync(join(cwd, 'LICENSE'), 'MIT');
    writeFileSync(join(cwd, 'CHANGELOG.md'), '# Release notes');
    for (const name of packages) {
      mkdirSync(join(cwd, 'packages', name), { recursive: true });
      writeFileSync(
        join(cwd, 'packages', name, 'package.json'),
        JSON.stringify({
          name,
          version: '0.1.0',
          files: ['dist'],
          ...(name.endsWith('trainer') ? { dependencies: { 'three-ntc': 'workspace:^', fflate: '^0.8.2' } } : {}),
        }),
      );
    }
    // Version bumping and the workspace:^ -> real semver rewrite are left to
    // @anolilab/semantic-release-pnpm / `pnpm pack`, not this script.
    prepare({}, { cwd, nextRelease: { version: '0.2.0' } });
    for (const name of packages) {
      const dir = join(cwd, 'packages', name);
      const pkg = JSON.parse(readFileSync(join(dir, 'package.json')));
      assert.equal(pkg.version, '0.1.0');
      assert.ok(pkg.files.includes('CHANGELOG.md'));
      assert.equal(readFileSync(join(dir, 'LICENSE'), 'utf8'), 'MIT');
      assert.equal(readFileSync(join(dir, 'CHANGELOG.md'), 'utf8'), '# Release notes');
      if (name.endsWith('trainer')) {
        assert.equal(pkg.dependencies['three-ntc'], 'workspace:^');
        assert.equal(pkg.dependencies.fflate, '^0.8.2');
      }
    }
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});
for (const [message, expected] of [
  ['fix: handle empty textures', 'patch'],
  ['feat: add exporter', 'minor'],
  ['feat!: replace the public loader API', 'major'],
  ['fix: change loader\n\nBREAKING CHANGE: remove the legacy API', 'major'],
  ['docs: explain installation', null],
  ['chore: update CI', null],
]) {
  test(`release level for ${message.split('\n')[0]}`, async () => {
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

test('the full pnpm prepare pipeline (files, per-package version bump, artifact pack) produces installable registry tarballs, with pnpm natively resolving workspace:^ to a real semver range', async () => {
  const { cpSync, existsSync } = await import('node:fs');
  const { execFileSync } = await import('node:child_process');
  const { prepare: pnpmPrepare } = await import('@anolilab/semantic-release-pnpm');
  const cwd = mkdtempSync(join(tmpdir(), 'ntc-pack-'));
  try {
    writeFileSync(join(cwd, 'LICENSE'), readFileSync('LICENSE'));
    writeFileSync(join(cwd, 'CHANGELOG.md'), '# Test release');
    // A real (if minimal) pnpm workspace, so `pnpm pack` can resolve the
    // trainer's `workspace:^` dependency on three-ntc the same way it does
    // in CI, instead of mocking that resolution.
    writeFileSync(join(cwd, 'pnpm-workspace.yaml'), 'packages:\n  - packages/*\n');
    writeFileSync(
      join(cwd, 'package.json'),
      JSON.stringify({ name: 'ntc-pack-test-root', version: '0.0.0', private: true }),
    );
    for (const name of packages) {
      const dir = join(cwd, 'packages', name);
      mkdirSync(dir, { recursive: true });
      cpSync(`packages/${name}/dist`, join(dir, 'dist'), { recursive: true });
      cpSync(`packages/${name}/package.json`, join(dir, 'package.json'));
      cpSync(`packages/${name}/README.md`, join(dir, 'README.md'));
    }
    const context = {
      cwd,
      env: process.env,
      nextRelease: { version: '0.2.0' },
      stdout: process.stdout,
      stderr: process.stderr,
      logger: { log() {} },
    };
    // Mirrors release.config.js's plugin order: files prep, then each
    // package's pnpm version bump, then our artifact pack step.
    prepare({}, context);
    execFileSync('pnpm', ['install', '--no-frozen-lockfile', '--ignore-scripts'], {
      cwd,
      stdio: 'pipe',
    });
    for (const name of packages) {
      await pnpmPrepare({ pkgRoot: `packages/${name}`, npmPublish: false }, context);
    }
    prepare({ artifacts: true }, context);
    for (const name of packages) {
      const tarball = join(cwd, 'release-artifacts', `${name}-0.2.0.tgz`);
      assert.ok(existsSync(tarball), `${name}: tarball not found at ${tarball}`);
      const entries = execFileSync('tar', ['-tf', tarball], { encoding: 'utf8' });
      for (const file of ['dist/index.js', 'dist/index.d.ts', 'LICENSE', 'README.md', 'CHANGELOG.md']) {
        assert.ok(entries.includes(`package/${file}`), `${name}: missing ${file}`);
      }
      const pkg = JSON.parse(execFileSync('tar', ['-xOf', tarball, 'package/package.json'], { encoding: 'utf8' }));
      assert.equal(pkg.version, '0.2.0');
      assert.ok(!JSON.stringify(pkg).includes('workspace:'));
      if (name.endsWith('trainer')) assert.equal(pkg.dependencies['three-ntc'], '^0.2.0');
    }
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test('renders release notes with the installed Conventional Commits preset', async () => {
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
