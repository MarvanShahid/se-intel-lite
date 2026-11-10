"""Deterministic metrics: ranking (P@k, nDCG@k), probabilistic (AUC, Brier, ECE), statistics."""
from __future__ import annotations

import math
from statistics import mean, pstdev

import numpy as np


def precision_at_k(ranked: list[str], relevant: dict[str, float], k: int = 10) -> float:
    top = ranked[:k]
    return sum(1 for x in top if relevant.get(x, 0) > 0) / k if k else 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, float], k: int = 10) -> float:
    """Graded nDCG; gain = log2(1 + n_fix_commits)."""
    def gain(x):
        return math.log2(1 + relevant.get(x, 0))
    dcg = sum(gain(x) / math.log2(i + 2) for i, x in enumerate(ranked[:k]))
    ideal = sorted((relevant.get(x, 0) for x in relevant), reverse=True)[:k]
    idcg = sum(math.log2(1 + g) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def roc_auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def bootstrap(fn, *arrays, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Mean and std of a metric over bootstrap resamples (skipping degenerate resamples)."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a) for a in arrays]
    m = len(arrs[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = fn(*[a[idx] for a in arrs])
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def mean_std(xs) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (mean(xs), pstdev(xs)) if xs else (float("nan"), float("nan"))


def wilcoxon(a, b) -> dict:
    """Paired two-sided Wilcoxon signed-rank test over questions."""
    from scipy.stats import wilcoxon as _w
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.allclose(a, b):
        return {"n": int(len(a)), "statistic": None, "p": None}
    r = _w(a, b, zero_method="wilcox")
    return {"n": int(len(a)), "statistic": float(r.statistic), "p": float(r.pvalue)}
