import { copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const packages = ["three-ntc", "three-ntc-trainer"];

// Only the CI checkout is modified. Git tags are the version source of truth.
export function prepare(_config, { cwd, nextRelease: { version } }) {
  for (const name of packages) {
    const dir = join(cwd, "packages", name);
    const path = join(dir, "package.json");
    const pkg = JSON.parse(readFileSync(path, "utf8"));
    pkg.version = version;
    if (pkg.dependencies?.["three-ntc"]) pkg.dependencies["three-ntc"] = `^${version}`;
    pkg.files = [...new Set([...pkg.files, "CHANGELOG.md"])];
    writeFileSync(path, `${JSON.stringify(pkg, null, 2)}\n`);
    copyFileSync(join(cwd, "LICENSE"), join(dir, "LICENSE"));
    copyFileSync(join(cwd, "CHANGELOG.md"), join(dir, "CHANGELOG.md"));
  }
}
