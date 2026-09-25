# Releases and repository setup

## npm trusted publishing

Configure **both** existing npm packages, `three-ntc` and `three-ntc-trainer`, at
npmjs.com → package → Settings → Trusted publishing → GitHub Actions:

| Field                     | Value         |
| ------------------------- | ------------- |
| Organization or user      | `bhouston`    |
| Repository                | `three-ntc`   |
| Workflow filename         | `release.yml` |
| Environment               | `npm`         |
| Allowed action (if shown) | `npm publish` |

The filename is case-sensitive and has no `.github/workflows/` prefix.
No npm token secret is needed. GitHub-hosted Ubuntu, Node 26, and
`id-token: write` provide OIDC; npm generates provenance automatically.
See [npm's trusted publishing documentation](https://docs.npmjs.com/trusted-publishers/).
The repository URL in both package manifests must match this repository.

## One-time bootstrap (historical)

- `main` and `dev` were created from the `master` tip, with `master` retained as
  history. `v0.1.0` was seeded at `f51b2304714ce8853f6bc5cb9293dc70825b4ff9`, the
  `gitHead` recorded on npm for both existing 0.1.0 packages, so semantic-release
  did not treat the project as an unreleased 1.0.0 package.
- Private vulnerability reporting is enabled in GitHub security settings.
- GitHub environment `npm` restricts deployments to `main`.
- `main` requires PRs, the `Unit` and `Issue, branch and commits` checks, blocks
  force pushes/deletion, and enforces for admins.
- The repository has since moved off the `dev` → `main` promotion model
  described in earlier revisions of this document; see "Routine release" below.
  `dev` is left inactive rather than deleted, in case any old references depend
  on it.

## Routine release

Contributors branch from `main` and open issue-linked PRs directly into `main`
(see `CONTRIBUTING.md`). Merging a PR runs CI only; nothing publishes
automatically.

When `main` has release-worthy commits ready to ship, the maintainer runs:

```sh
gh workflow run release.yml --ref main
```

The workflow rejects any dispatch not targeting `refs/heads/main`, re-runs the
required quality gates against the exact commit that was dispatched (the
separate PR/push CI run retains the advisory GPU suite), aborts if `main` has
advanced since the dispatch instead of releasing an unvalidated commit,
restores the previous release changelog, and runs semantic-release. Only
release-worthy commits produce a version; a dispatch with none is a successful
no-op reported in the run summary. Pass `dry_run: true` to validate changelog
rendering and tag ancestry without publishing, tagging, or creating a GitHub
release.

The `npm` environment and workflow permissions apply only to the publish job.
PR and push CI jobs have read-only access. The workflow never uses a
long-lived npm token. GitHub release assets retain the generated changelog
between releases. Ordinary PR merges and tag pushes never trigger publishing.

Two registry publications are sequential, not atomic. If publishing fails after
one package succeeds, do not blindly rerun: inspect the npm versions, workflow
logs and Git tag. Recover the missing package/release from the recorded build
and version, or prepare a corrective release; an existing npm version cannot
be overwritten. Do not delete a published version to repair a CI run.

## Two-package release

Both npm packages, `three-ntc` and `three-ntc-trainer`, share one semantic-release
version and changelog. The trainer is published with a registry dependency on that
runtime version, not a workspace link. `v*` Git tags are authoritative; checked-in
manifests are development placeholders. Generated `CHANGELOG.md` and npm tarballs are
attached to each GitHub release, and the changelog is included in each published
package. No bot commits or branch protection bypass are needed.

## Validation and rollout

The full dependency audit currently reports two high-severity `extract-zip@2.0.1`
advisories through WebdriverIO/Puppeteer. The registry currently has no 2.0.2
release; full audit is advisory, while production audit blocks CI. Upgrade and
make the full audit blocking when a compatible patched version is available.

`pnpm release:check` tests version analysis (including `!` and breaking footers),
release document preparation, and pnpm's native `workspace:` resolution without
publishing. `pnpm --dir <pkgRoot> pack --dry-run` can inspect prepared artifacts
in an isolated checkout (plain `pnpm pack <dir>` ignores a directory argument;
`--dir` is required to target a specific workspace package).
Never run the real release command as a local packaging test.

Coverage is uploaded as a CI artifact and summarized in checks; the README badge
is updated from successful `main` CI on the dedicated `coverage-badge` branch.
After this pilot's PR and first OIDC release succeed, extract the shared files
into a separate template repo and add a copy script there. Review project-specific
names, branching, package topology and limits for every rollout.
