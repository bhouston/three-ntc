import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const { pull_request: pr } = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, "utf8"));
if (pr.base.ref !== "main") throw new Error("Open PRs against main.");
const match = /^(?:feature|fix|docs|chore|refactor|test|perf|ci)\/(\d+)-[a-z0-9-]+$/.exec(
  pr.head.ref,
);
if (!match) throw new Error("Use a branch such as feature/42-batch-export.");
const closes = new RegExp(`\\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\s+#${match[1]}\\b`, "i");
if (!closes.test(pr.body ?? "")) throw new Error(`PR body must include Closes #${match[1]}.`);
execFileSync("pnpm", ["exec", "commitlint"], {
  input: pr.title,
  stdio: ["pipe", "inherit", "inherit"],
});
execFileSync("pnpm", ["exec", "commitlint", "--from", pr.base.sha, "--to", pr.head.sha], {
  stdio: "inherit",
});
