# three-ntc documentation site

The Docusaurus site is a static sibling of the legacy TanStack Start demo. The
legacy `packages/website` and its Cloud Run deployment remain available.

From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm docs:build
pnpm --filter ntc-docs serve
```

`docs:build` builds the runtime and trainer, the standalone Vite playground,
then generates API reference pages with TypeDoc and builds Docusaurus. API pages
come directly from each published package's `src/index.ts` exports and JSDoc;
they are ignored build artifacts. Edit source comments rather than generated pages.
TypeDoc uses TypeScript 5.9; the library is checked independently with the workspace
compiler. Authored guides are in `docs/`.

The playground's production build is copied to `build/tools/` after Docusaurus
builds, including an ES module and stylesheet for direct React mounting. The
Docusaurus `ToolEmbed` component mounts the local tools in a shadow root to
isolate their styles; there are no iframes or requests to the old website.
Embedded tools omit the standalone navigation. Assets resolve relative to the
module, so they work under a project subpath. `pnpm docs:dev` stages the tools in the
ignored `static/tools/` directory before serving authored documentation. For
interactive tool hot reload, use `pnpm --filter ntc-playground dev` separately.

The default deployment is <https://three-ntc.ben3d.ca/>. Set `SITE_URL`
and `BASE_URL` when building for a different origin/path.

`.github/workflows/pages.yml` deploys the site only as part of an official release:
`release.yml` calls it with the new `v*` tag, and it builds from that tag. It never runs
on merges to main. Dispatch it manually (optionally with a `ref` tag; default is the
latest `v*` tag) to redeploy a release. Only tags that contain `packages/docs` can be
built; `v0.2.0` predates the site. The repository's Pages build source must be
**GitHub Actions**. The initial deployment was published from a `gh-pages` snapshot
and stays live until the first release deploys through the workflow.

The trainer requires WebGPU and a secure context. GPU visual checks must use a
physical adapter; software renderers do not count as validation.
