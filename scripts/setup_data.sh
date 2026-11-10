#!/usr/bin/env sh
# Clone the frozen evaluation repositories at their pinned commits into ./repos and the SWE-QA benchmark into
# ./data/SWE-QA-Bench (pinned commit). Usage: scripts/setup_data.sh [repo ...]   (default: all four)
# Full history is kept on purpose: Gap 2/3 mine bug-fix commits from it.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p repos data
if [ ! -d data/SWE-QA-Bench ]; then
  git clone -q https://github.com/peng-weihan/SWE-QA-Bench data/SWE-QA-Bench
fi
git -C data/SWE-QA-Bench checkout -q c13deac7a0d99b0ca2e593e004c4739475785b08
$PY - "$@" <<'PY'
import os, subprocess, sys, yaml
cfg = yaml.safe_load(open("configs/repos.yaml"))
want = set(sys.argv[1:])
for r in cfg["repos"]:
    if want and r["name"] not in want:
        continue
    d = f"repos/{r['name']}"
    if not os.path.isdir(d):
        print(f"cloning {r['name']} ..."); subprocess.run(["git", "clone", "-q", r["url"], d], check=True)
    subprocess.run(["git", "-C", d, "checkout", "-q", r["commit"]], check=True)
    n = subprocess.run(["git", "-C", d, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
    print(f"{r['name']}@{r['commit']}: {len(n)} tracked .py files (config says {r['py_files']})")
PY
