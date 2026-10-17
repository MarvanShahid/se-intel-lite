#!/usr/bin/env sh
# Clone the frozen evaluation repositories at their pinned commits into ./repos
# Usage: scripts/setup_data.sh [repo ...]     (default: all four)
# Full history is kept on purpose: Phase 3/4 mine bug-fix commits with PyDriller.
set -e
cd "$(dirname "$0")/.."
mkdir -p repos
python3 - "$@" <<'PY'
import subprocess, sys, yaml
cfg = yaml.safe_load(open("configs/repos.yaml"))
want = set(sys.argv[1:])
for r in cfg["repos"]:
    if want and r["name"] not in want:
        continue
    d = f"repos/{r['name']}"
    import os
    if not os.path.isdir(d):
        print(f"cloning {r['name']} ..."); subprocess.run(["git", "clone", "-q", r["url"], d], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", r["commit"]], check=True)
    n = subprocess.run(["git", "-C", d, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
    print(f"{r['name']}@{r['commit']}: {len(n)} tracked .py files (config says {r['py_files']})")
PY
