# Contributing

This is the shared workflow for human contributors, Claude, and Codex.
Read this file before starting a task. Do not commit directly to `main` or `dev`.

1. Before implementing a feature or fix, create a GitHub issue (or reuse the
   supplied issue). Follow the feature template: description, motivation,
   constraints, acceptance criteria. Agents may use `gh issue create`.
2. Branch from current `origin/dev` using `<type>/<issue>-<short-description>`,
   for example `feature/42-batch-export`. Allowed branch types are `feature`,
   `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, and `ci`.
3. Use Conventional Commits for every commit and the PR title:
   `type(optional-scope): description`. Types include `feat`, `fix`, `perf`,
   `docs`, `chore`, `refactor`, `test`, `style`, `build`, `ci`, and `revert`.
   Reference the issue in the commit body when useful. `feat` releases a minor,
   `fix` and `perf` a patch, and `!` or a `BREAKING CHANGE:` footer a major.
   Other types normally do not release. Do not manually bump package versions
   or edit generated release notes. Husky runs commitlint on each local commit;
   CI checks all implementation commits and the PR title.
4. Install with `pnpm install --frozen-lockfile` using the Node version in
   `.nvmrc` and pnpm version in `package.json`. Run `pnpm build`, `pnpm lint`,
   `pnpm test`, `pnpm size`, `pnpm release:check`, and
   `pnpm audit --prod --audit-level high`. Run `pnpm test:gpu` for GPU changes;
   hosted SwiftShader GPU checks are advisory.
5. Push the branch and open a PR **against `dev`** with `Closes #<issue>` and
   validation results. Use the PR template. Keep each feature PR focused.
   Squash feature PRs using the validated Conventional Commit title, or retain
   their validated commits with a merge commit. Do not bypass required checks.
6. For a release, open `dev` → `main` and use a **merge commit**, never squash or
   rebase this PR: preserving ancestry preserves release history. Only pushes
   to `main` run semantic-release. After release, open a `main` → `dev` sync PR and merge it with a merge
   commit to bring release ancestry back into the integration branch. GitHub closing keywords close issues when they reach the default
   branch (`main`), not when the implementation first lands on `dev`.

Unit coverage includes **all runtime and trainer TypeScript source**, including
GPU paths that unit tests do not execute. Initial gates are statements 19%,
branches 13%, functions 18%, lines 19%, based on measured coverage rather than an
artificial 80% promise. Raise gates as CPU/GPU coverage improves; do not lower
or exclude files just to pass. CI uploads HTML/LCOV reports and a summary.
Bundle limits measure minified gzip ESM exports with three.js external:
11,000 bytes runtime and 52,000 bytes trainer. Explain intentional limit changes.
High/critical production dependency advisories block CI. The full development
audit is visible but advisory while extract-zip has no patched registry release.

Both npm packages share one semantic-release version and changelog. The trainer
is published with a registry dependency on that runtime version. `v*` Git tags
are authoritative; checked-in manifests remain development placeholders.
Generated `CHANGELOG.md` and npm tarballs are attached to each GitHub release,
and the changelog is included in each package. No bot commits or branch
protection bypass are needed. See [release setup](docs/releasing.md).

For another repository, copy this standard, the two agent pointer files,
commitlint/Husky configuration and issue/PR templates. Adapt package names,
release preparation, coverage and size limits before copying workflows. Validate
a real issue/PR cycle before extracting a separate GitHub template repository;
this repository's monorepo release configuration is not a universal template.
