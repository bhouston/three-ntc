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

The default deployment is <https://bhouston.github.io/three-ntc/>. Set `SITE_URL`
and `BASE_URL` when building for a different origin/path. No custom domain is
claimed, so the old website continues serving its existing domain.

`.github/workflows/pages.yml` builds and deploys on pushes to main and manual
dispatches. The repository's Pages build source must be **GitHub Actions** when
using this workflow. The initial deployment was published from a `gh-pages`
snapshot; the repository is then configured for GitHub Actions so the workflow
takes over after merge. No merge is required to view the initial deployment.

The trainer requires WebGPU and a secure context. GPU visual checks must use a
physical adapter; software renderers do not count as validation.
