export default {
  branches: ["main"],
  tagFormat: "v${version}",
  plugins: [
    ["@semantic-release/commit-analyzer", { preset: "conventionalcommits" }],
    ["@semantic-release/release-notes-generator", { preset: "conventionalcommits" }],
    ["@semantic-release/changelog", { changelogFile: "CHANGELOG.md" }],
    "./scripts/prepare-release.mjs",
    ["@anolilab/semantic-release-pnpm", { pkgRoot: "packages/three-ntc" }],
    ["@anolilab/semantic-release-pnpm", { pkgRoot: "packages/three-ntc-trainer" }],
    // Packs both (now version-bumped) packages into release-artifacts/ for
    // the GitHub release below. See the comment in prepare-release.mjs for
    // why this isn't done via @anolilab/semantic-release-pnpm's own
    // `tarballDir` option.
    ["./scripts/prepare-release.mjs", { artifacts: true }],
    [
      "@semantic-release/github",
      {
        successComment: false,
        failComment: false,
        releasedLabels: false,
        assets: ["release-artifacts/*.tgz", "CHANGELOG.md"],
      },
    ],
  ],
};
