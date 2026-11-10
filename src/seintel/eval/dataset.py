"""Config, SWE-QA question loading, fixed dev/test split and deterministic question attributes."""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SPLIT_SEED = 13           # fixed once in Phase 1; never changed
DEV_PER_REPO = 5          # 4 repos x 5 = 20 development questions; the other 172 are the test set


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs" / "repos.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(name: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg["repos_dir"] / name


def _bench_file(name: str, cfg: dict) -> Path:
    for base in (ROOT / cfg["benchmark_dir"], ROOT / "data" / "SWE-QA-Bench" / "Benchmark"):
        p = base / f"{name}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(f"SWE-QA file for {name} not found; run scripts/setup_data.sh")


@lru_cache(maxsize=None)
def _load_repo_questions(name: str) -> tuple[dict, ...]:
    cfg = load_config()
    rows = []
    with open(_bench_file(name, cfg), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                d = json.loads(line)
                rows.append({"qid": f"{name}-{i:03d}", "repo": name, "question": d["question"],
                             "answer": d["answer"], "qtype": question_type(d["question"])})
    return tuple(rows)


def split_ids(name: str) -> dict[str, list[str]]:
    ids = [q["qid"] for q in _load_repo_questions(name)]
    rnd = random.Random(f"{SPLIT_SEED}-{name}")
    dev = sorted(rnd.sample(ids, DEV_PER_REPO))
    return {"dev": dev, "test": [i for i in ids if i not in dev]}


def load_questions(split: str = "test", repos: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """split: dev | test | all. `limit` keeps the first n per repo (for smoke runs only)."""
    cfg = load_config()
    out = []
    for r in cfg["repos"]:
        if repos and r["name"] not in repos:
            continue
        qs = list(_load_repo_questions(r["name"]))
        if split != "all":
            keep = set(split_ids(r["name"])[split])
            qs = [q for q in qs if q["qid"] in keep]
        out.extend(qs[:limit] if limit else qs)
    return out


def question_type(q: str) -> str:
    w = q.strip().split()[0].lower().strip("?,") if q.strip() else ""
    return w if w in ("what", "why", "where", "how") else "other"


_IDENT = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`|\b([A-Za-z_]\w*\.py)\b")


def reference_entities(answer: str, rg) -> set[str]:
    """Graph entities named in a reference answer (backticked identifiers and file names)."""
    names = {}
    for nid, d in rg.entities():
        names.setdefault(d["name"], []).append(nid)
    found = set()
    for m in _IDENT.finditer(answer):
        tok = m.group(1) or m.group(2)
        last = tok.split(".")[-1] if not tok.endswith(".py") else tok.rsplit("/", 1)[-1]
        if tok.endswith(".py"):
            found.update(n for n in names.get(last, []) if rg.node(n)["type"] == "file")
            continue
        cands = names.get(last, [])
        if 0 < len(cands) <= 3:          # ambiguous common names (e.g. `get`) are ignored
            found.update(cands)
    return found


def is_multi_hop(answer: str, rg) -> bool:
    """Deterministic multi-hop label: the reference answer relies on entities from >= 2 different files."""
    files = {rg.file_of(e) for e in reference_entities(answer, rg)}
    files.discard(None)
    return len(files) >= 2
