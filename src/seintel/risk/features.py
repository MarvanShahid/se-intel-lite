"""Per-file feature/label tables at a time-based snapshot (shared by Gap 2 hotspots and Gap 3 risk)."""
from __future__ import annotations

import json
from pathlib import Path

from ..eval.dataset import ROOT
from ..graph.store import RepoGraph
from ..history.mining import (extract_snapshot, files_at, fix_labels, history_features, is_source_file,
                              load_history, make_splits)
from ..quality.metrics import graph_features, static_metrics

STATIC = ["loc", "sloc", "cc_sum", "cc_max", "cc_mean", "mi", "n_blocks"]
CHANGE = ["n_commits", "churn", "n_authors", "prior_fixes", "recent_commits", "age_days", "days_since_change"]
GRAPH = ["fan_in", "fan_out", "pagerank", "dependents_2hop", "calls_in"]

CACHE = ROOT / ".cache"


def snapshot_table(repo_dir: str | Path, sha: str, upto: int, commits) -> list[dict]:
    """Feature rows for every non-test source file that exists at commit `sha`."""
    snap = extract_snapshot(repo_dir, sha, CACHE / "snapshots")
    paths = {p for p in files_at(repo_dir, sha) if is_source_file(p)}
    rg = RepoGraph.from_repo(snap, cache_dir=CACHE / "graphs" / Path(repo_dir).name / sha[:12])
    gf = graph_features(rg)
    hf = history_features(commits, upto, paths)
    rows = []
    for p in sorted(paths):
        src = "\n".join(rg.lines(p))
        sm = static_metrics(src)
        g = gf.get(p, {k: 0 for k in GRAPH})
        rows.append({"path": p, **{k: sm.get(k, 0) for k in STATIC}, **hf[p], **g})
    return rows


def build_split_table(repo: str, split: str, refresh: bool = False) -> list[dict]:
    """Rows for one repository and split (train|tune|test) with labels `n_fixes` and `y` (any fix)."""
    out_f = CACHE / "tables" / f"{repo}.{split}.json"
    if out_f.exists() and not refresh:
        return json.loads(out_f.read_text())
    repo_dir = ROOT / "repos" / repo
    commits = load_history(str(repo_dir))
    sp = make_splits(len(commits))[split]
    sha = commits[sp.snapshot].sha
    rows = snapshot_table(repo_dir, sha, sp.snapshot, commits)
    labels = fix_labels(commits, sp, {r["path"] for r in rows})
    for r in rows:
        r.update(repo=repo, split=split, snapshot=sha[:12], n_fixes=labels[r["path"]], y=int(labels[r["path"]] > 0))
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(rows))
    return rows


def build_current_table(repo: str) -> list[dict]:
    """Rows at the pinned commit (HEAD of the checkout); no labels. Used by the agent tools."""
    out_f = CACHE / "tables" / f"{repo}.current.json"
    if out_f.exists():
        return json.loads(out_f.read_text())
    repo_dir = ROOT / "repos" / repo
    commits = load_history(str(repo_dir))
    rows = snapshot_table(repo_dir, commits[-1].sha, len(commits) - 1, commits)
    for r in rows:
        r.update(repo=repo, split="current", snapshot=commits[-1].sha[:12])
    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(rows))
    return rows


def split_info(repo: str) -> dict:
    commits = load_history(str(ROOT / "repos" / repo))
    from datetime import datetime, timezone
    info = {"n_commits": len(commits), "n_bugfix_commits": sum(c.bugfix for c in commits)}
    for k, sp in make_splits(len(commits)).items():
        a, b = commits[sp.snapshot], commits[sp.end]
        info[k] = {"snapshot": a.sha[:12],
                   "snapshot_date": datetime.fromtimestamp(a.ts, timezone.utc).date().isoformat(),
                   "window_end_date": datetime.fromtimestamp(b.ts, timezone.utc).date().isoformat(),
                   "window_commits": sp.end - sp.snapshot,
                   "window_fix_commits": sum(c.bugfix for c in commits[sp.snapshot + 1: sp.end + 1])}
    return info
