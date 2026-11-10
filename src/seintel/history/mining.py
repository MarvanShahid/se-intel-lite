"""Git history mining for Gap 2 (hotspot ground truth) and Gap 3 (risk features and labels).

We use the git CLI directly (one `git log --numstat` call per repository) instead of PyDriller: it is an order of
magnitude faster on xarray/pytest and gives the same commit/file/churn information we need.

Bug-fix commits are found with message keywords (Phase 1 B.7). The rules below were fixed on the development
repository `requests` before any test-window number was computed; label noise is estimated by a manual check of
50 sampled commits (scripts/sample_fix_commits.py writes the sample to label).
"""
from __future__ import annotations

import io
import re
import subprocess
import tarfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

FIX_RE = re.compile(r"\b(fix|fixes|fixed|fixing|bugfix|bug|bugs|defect|regression|crash(es|ed)?|hotfix)\b", re.I)
REF_RE = re.compile(r"\b(close[sd]?|resolve[sd]?|fix(e[sd])?)\s+(gh-|#)\d+", re.I)
# "fix typo", "fix docs" etc. are not defect fixes
NOT_FIX_RE = re.compile(r"\b(typos?|docs?|documentation|docstrings?|changelog|readme|spelling|grammar|whitespace|"
                        r"lint(ing)?|flake8|mypy|formatting|wording|links?|ci|tox|pre-commit|test name)\b", re.I)


def is_bugfix(message: str) -> bool:
    subject = message.strip().splitlines()[0] if message.strip() else ""
    if subject.lower().startswith(("merge ", "revert \"merge", "release ", "bump ")):
        return False
    if REF_RE.search(message):
        return True
    if FIX_RE.search(subject):
        return not NOT_FIX_RE.search(subject)
    return False


def is_source_file(path: str) -> bool:
    """Non-test, non-doc Python source (the unit for hotspots and risk)."""
    if not path.endswith(".py"):
        return False
    p = "/" + path.lower()
    name = p.rsplit("/", 1)[-1]
    if any(s in p for s in ("/test/", "/tests/", "/testing/", "/doc/", "/docs/", "/examples/", "/example/",
                            "/benchmarks/", "/asv_bench/", "/ci/", "/.github/",
                            # vendored third-party code (e.g. requests/packages/urllib3 before 2017)
                            "/packages/", "/vendor/", "/_vendor/", "/vendored/", "/extern/", "/_vendored/")):
        return False
    return not (name.startswith("test_") or name.endswith("_test.py") or name in ("conftest.py", "setup.py",
                                                                                   "noxfile.py", "conf.py"))


@dataclass
class Commit:
    sha: str
    ts: int
    author: str
    message: str
    files: dict = field(default_factory=dict)  # path -> (added, deleted)
    bugfix: bool = False


def _git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8", errors="replace")


@lru_cache(maxsize=16)
def load_history(repo: str | Path, until: str = "HEAD") -> tuple[Commit, ...]:
    """All non-merge commits reachable from `until`, oldest first."""
    repo = Path(repo)
    sep, end = "\x1f", "\x1e"
    out = _git(repo, "log", until, "--no-merges", "--no-renames", "--reverse", "--numstat",
               f"--format={end}%H{sep}%at{sep}%aN{sep}%B{sep}")
    commits = []
    for block in out.split(end)[1:]:
        sha, ts, author, msg, rest = block.split(sep, 4)
        c = Commit(sha, int(ts), author, msg.strip())
        for line in rest.strip().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[2].endswith(".py"):
                a = int(parts[0]) if parts[0].isdigit() else 0
                d = int(parts[1]) if parts[1].isdigit() else 0
                c.files[parts[2]] = (a, d)
        c.bugfix = is_bugfix(c.message)
        commits.append(c)
    return tuple(commits)


@dataclass
class Split:
    """Time-based split on the commit sequence (fractions of the commit count, Phase 1 B.11)."""
    name: str
    snapshot: int   # index of the snapshot commit (features use commits[:snapshot+1])
    end: int        # labels use commits[snapshot+1 : end+1]


SPLIT_FRACTIONS = {"train": (0.40, 0.60), "tune": (0.60, 0.75), "test": (0.75, 1.00)}


def make_splits(n_commits: int, fractions=None) -> dict[str, Split]:
    fr = fractions or SPLIT_FRACTIONS
    return {k: Split(k, max(0, int(a * n_commits) - 1), max(0, int(b * n_commits) - 1)) for k, (a, b) in fr.items()}


def files_at(repo: str | Path, sha: str) -> list[str]:
    return [p for p in _git(Path(repo), "ls-tree", "-r", "--name-only", sha).splitlines() if p.endswith(".py")]


def extract_snapshot(repo: str | Path, sha: str, cache_root: str | Path) -> Path:
    """Write the .py files of commit `sha` to cache_root/<repo>/<sha>/ (idempotent) and return that path."""
    repo = Path(repo)
    dest = Path(cache_root) / repo.name / sha[:12]
    done = dest / ".complete"
    if done.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    data = _git(repo, "archive", "--format=tar", sha, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".py")]
        for m in members:
            target = dest / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(m)
            if f is not None:
                target.write_bytes(f.read())
    done.write_text(sha)
    return dest


def history_features(commits, upto: int, paths: set[str]) -> dict[str, dict]:
    """Change features per path using commits[:upto+1] only (no look-ahead)."""
    now = commits[upto].ts
    f = {p: {"n_commits": 0, "churn_added": 0, "churn_deleted": 0, "authors": set(), "prior_fixes": 0,
             "recent_commits": 0, "first_ts": None, "last_ts": None} for p in paths}
    recent_from = now - 365 * 86400
    for c in commits[: upto + 1]:
        for p, (a, d) in c.files.items():
            if p not in f:
                continue
            x = f[p]
            x["n_commits"] += 1
            x["churn_added"] += a
            x["churn_deleted"] += d
            x["authors"].add(c.author)
            x["prior_fixes"] += c.bugfix
            x["recent_commits"] += c.ts >= recent_from
            x["first_ts"] = x["first_ts"] or c.ts
            x["last_ts"] = c.ts
    out = {}
    for p, x in f.items():
        out[p] = {"n_commits": x["n_commits"], "churn": x["churn_added"] + x["churn_deleted"],
                  "n_authors": len(x["authors"]), "prior_fixes": x["prior_fixes"],
                  "recent_commits": x["recent_commits"],
                  "age_days": (now - x["first_ts"]) / 86400 if x["first_ts"] else 0.0,
                  "days_since_change": (now - x["last_ts"]) / 86400 if x["last_ts"] else 9999.0}
    return out


def fix_labels(commits, split: Split, paths: set[str]) -> dict[str, int]:
    """Number of bug-fix commits touching each path in the label window (snapshot, end]."""
    lab = {p: 0 for p in paths}
    for c in commits[split.snapshot + 1: split.end + 1]:
        if c.bugfix:
            for p in c.files:
                if p in lab:
                    lab[p] += 1
    return lab
