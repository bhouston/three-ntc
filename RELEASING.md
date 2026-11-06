# Releasing

Releases are manual and maintainer-only. Merging to `main` never publishes.

## Running a Release

```sh
gh workflow run release.yml --ref main                  # release
gh workflow run release.yml --ref main -f dry_run=true  # preview only, publishes nothing
```

When there are no release-worthy commits since the last `v*` tag, the run is a no-op.

## What the Workflow Does

1. Refuses any ref other than `main`.
2. Runs the full CI workflow on the dispatched commit.
3. Stops if `main` has moved since the dispatch.
4. Installs, builds, and runs `pnpm release`, which is semantic-release.
5. semantic-release computes the next version, creates the `v<version>` tag and GitHub Release, and publishes.

Releases run one at a time and never cancel a publish in progress. npm packages publish through npm trusted publishing (GitHub OIDC), so no `NPM_TOKEN` is stored.

## Versioning

The version comes from the Conventional Commits since the last `v*` tag. The highest change wins:

- `feat` produces a minor release.
- `fix` and `perf` produce a patch release.
- `!` or a `BREAKING CHANGE:` footer produces a major release.
- Other types do not trigger a release.

Everything this repository publishes shares one version. `package.json` versions in source are development snapshots; the Git tag and the registry hold the released version. [GitHub Releases](https://github.com/bhouston/three-ntc/releases) is the changelog of record.

## One-Time Setup

1. On npmjs.com, open each package's Trusted Publisher settings, choose GitHub Actions, and enter organization or user `bhouston`, repository `three-ntc`, workflow filename `release.yml`, and environment name `npm`.
2. Protect `main`: require pull requests and the CI checks, allow merge commits, and block force pushes and deletion.
3. If the repository uploads coverage to Codecov, add the `CODECOV_TOKEN` repository secret.

## Recovery

Publishing is not atomic. If a run fails part way, compare the registry versions, the `v<version>` tag, and the workflow log, then finish what is missing from the same release commit. If publishing succeeded but the GitHub Release failed, create the release from the existing tag. Never republish a version, delete a published version, or move a release tag. Resolve the failure before dispatching another release.

## Repository Specifics

### Published Artifacts

- npm packages `three-ntc` (runtime) and `three-ntc-trainer`, published in that order and sharing one version. Each GitHub Release also attaches the two npm tarballs.
- The documentation site, built from the release tag and deployed to GitHub Pages.

The trainer is published with a registry dependency on the matching runtime version, not a workspace link. Trusted publishing must be configured for both packages, and the repository URL in both manifests must match this repository.

### Website

After a real release (never a dry run), `release.yml` calls `pages.yml` with the new tag, which builds and deploys the documentation site from that tag. Merges to `main` never deploy it. To redeploy, dispatch `pages.yml` with an optional `ref`, which defaults to the latest `v*` tag; the tag must contain `packages/docs`.

### Local Validation

`pnpm release:check` tests version analysis (including `!` and breaking footers), release preparation, and pnpm's native `workspace:` resolution without publishing. Use `pnpm --dir <pkgRoot> pack --dry-run` to inspect a package; plain `pnpm pack <dir>` ignores the directory argument. Never run the real release command as a local packaging test.
