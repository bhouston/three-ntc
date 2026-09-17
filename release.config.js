export default {
  branches: ["main"],
  tagFormat: "v${version}",
  plugins: [
    ["@semantic-release/commit-analyzer", { preset: "conventionalcommits" }],
    ["@semantic-release/release-notes-generator", { preset: "conventionalcommits" }],
    ["@semantic-release/changelog", { changelogFile: "CHANGELOG.md" }],
    "./scripts/prepare-release.mjs",
    ["@semantic-release/npm", { pkgRoot: "packages/three-ntc", tarballDir: "release-artifacts" }],
    [
      "@semantic-release/npm",
      { pkgRoot: "packages/three-ntc-trainer", tarballDir: "release-artifacts" },
    ],
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
