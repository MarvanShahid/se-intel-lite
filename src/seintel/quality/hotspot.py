"""Gap 2 component: graph-aware hotspot ranking.

score(f) = z(log1p cc_sum) + w_in * z(log1p fan_in) + w_out * z(log1p fan_out)

z is computed within one repository snapshot. The weights (w_in, w_out) are chosen on the TUNE split only
(snapshot at 60% of history, ground truth = bug-fix commits in the 60-75% window); the TEST split
(snapshot at 75%, window 75-100%) is used once for the reported numbers.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..eval.metrics import ndcg_at_k, precision_at_k

WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_WEIGHTS = (1.0, 0.5)   # replaced by the tuned weights saved in results/gap2_weights.json


def _z(xs) -> np.ndarray:
    a = np.log1p(np.asarray(xs, float))
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a * 0


def score_rows(rows: list[dict], method: str, weights=DEFAULT_WEIGHTS) -> list[float]:
    if method == "loc":
        return [r["sloc"] for r in rows]
    if method == "cc":
        return [r["cc_sum"] for r in rows]
    if method == "churn":
        return [r["churn"] for r in rows]
    if method == "churn_x_cc":                 # classic change-complexity hotspot (reference only)
        return [math.log1p(r["churn"]) * math.log1p(r["cc_sum"]) for r in rows]
    if method == "graph_only":
        return list(_z([r["fan_in"] for r in rows]) + _z([r["fan_out"] for r in rows]))
    if method == "hotspot":
        w_in, w_out = weights
        return list(_z([r["cc_sum"] for r in rows]) + w_in * _z([r["fan_in"] for r in rows])
                    + w_out * _z([r["fan_out"] for r in rows]))
    raise ValueError(method)


def rank(rows: list[dict], method: str, weights=DEFAULT_WEIGHTS) -> list[str]:
    s = score_rows(rows, method, weights)
    order = sorted(range(len(rows)), key=lambda i: (-s[i], rows[i]["path"]))   # deterministic ties
    return [rows[i]["path"] for i in order]


def evaluate(rows: list[dict], method: str, weights=DEFAULT_WEIGHTS, k: int = 10) -> dict:
    ranked = rank(rows, method, weights)
    rel = {r["path"]: r["n_fixes"] for r in rows}
    return {"p@10": precision_at_k(ranked, rel, k), "ndcg@10": ndcg_at_k(ranked, rel, k)}


def tune_weights(tables_by_repo: dict[str, list[dict]], metric: str = "ndcg@10") -> tuple[tuple[float, float], list]:
    """Grid search on the tune split; ties broken towards smaller weights (simpler model)."""
    trials = []
    for w in itertools.product(WEIGHT_GRID, WEIGHT_GRID):
        vals = [evaluate(rows, "hotspot", w)[metric] for rows in tables_by_repo.values()]
        trials.append((float(np.mean(vals)), -sum(w), w))
    trials.sort(reverse=True)
    return trials[0][2], [{"w_in": t[2][0], "w_out": t[2][1], metric: t[0]} for t in trials]


def explain(row: dict, rank_pos: int, n: int) -> str:
    return (f"{row['path']} is hotspot #{rank_pos}/{n}: cyclomatic complexity sum={row['cc_sum']} "
            f"(max {row['cc_max']}), sloc={row['sloc']}, maintainability index={row['mi']}, "
            f"imported/used by {row['fan_in']} files, depends on {row['fan_out']} files, "
            f"{row['dependents_2hop']} files within 2 dependency hops.")
