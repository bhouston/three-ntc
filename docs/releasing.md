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

## One-time bootstrap

- Create `main` and `dev` from the current `master` tip; retain `master` as history.
  Make `main` the default branch. Feature PRs explicitly target `dev`.
- Seed `v0.1.0` at `f51b2304714ce8853f6bc5cb9293dc70825b4ff9`, the `gitHead`
  recorded on npm for **both** existing 0.1.0 packages. This prevents semantic-release
  from treating the project as an unreleased 1.0.0 package.
- Enable private vulnerability reporting in GitHub security settings.
- Create GitHub environment `npm`, restricting deployments to `main`.
- Protect `dev` and `main`: require PRs, `Unit` and `Issue, branch and commits` checks,
  prevent force pushes/deletion, and enforce for admins. The release PR policy allows
  only this repository's `dev` branch into `main`. Leave merge commits enabled for
  release PRs; squash only implementation PRs.
- Merge the setup PR into `dev`. Configure npm trust before merging `dev` into `main`.

## Routine release

Open a PR from `dev` to `main`; review CI, then **merge commit** it. The release
workflow runs the same quality gates again, builds, restores the previous
release changelog, and runs semantic-release. Only release-worthy commits
produce a version. Chore/docs-only changes may deploy the demo without an npm
release. The first workflow setup is a chore and does not itself request a bump.

The `npm` environment and workflow permissions apply only to the publish job.
PR jobs have read-only access. The workflow never uses a long-lived npm token.
GitHub release assets retain the generated changelog between releases.

Two registry publications are sequential, not atomic. If publishing fails after
one package succeeds, do not blindly rerun: inspect the npm versions, workflow
logs and Git tag. Recover the missing package/release from the recorded build
and version, or prepare a corrective release; an existing npm version cannot
be overwritten. Do not delete a published version to repair a CI run.

## Validation and rollout

The full dependency audit currently reports two high-severity `extract-zip@2.0.1`
advisories through WebdriverIO/Puppeteer. The registry currently has no 2.0.2
release; full audit is advisory, while production audit blocks CI. Upgrade and
make the full audit blocking when a compatible patched version is available.

`pnpm release:check` tests version analysis (including `!` and breaking footers),
shared version preparation, and workspace dependency conversion without publishing.
`npm pack --dry-run` can inspect prepared artifacts in an isolated checkout.
Never run the real release command as a local packaging test.

Coverage is uploaded as a CI artifact and summarized in checks; the README badge
is updated from successful `dev` CI on the dedicated `coverage-badge` branch.
Before first successful dev CI it may display unavailable.
After this pilot's PR and first OIDC release succeed, extract the shared files
into a separate template repo and add a copy script there. Review project-specific
names, branching, package topology and limits for every rollout.
