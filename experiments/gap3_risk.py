"""E4 (Gap 3) + ablations A4 (no graph features) and A5 (no calibration).

Hypothesis (Phase 1 B.10): the calibrated graph-aware model has a lower Brier score than the metrics-only model
on at least 3 of 4 repositories, with ECE <= 0.10.

Protocol: pooled training on the TRAIN split (snapshot at 40% of history, label window 40-60%), Platt calibration
on the TUNE split (snapshot 60%, window 60-75%), evaluation per repository on the TEST split (snapshot 75%,
window 75-100%). Deterministic, so one run; uncertainty via 1000 bootstrap resamples of test files.

Optional: --llm anthropic:<model> adds the "LLM-only risk guess" comparison (the LLM sees the file's code
metrics summary and path, no trained model) on the test split.

Usage: python experiments/gap3_risk.py [--llm anthropic:claude-opus-5-5]
"""
import argparse
import json
import re

import numpy as np
from _common import REPOS, ROOT, write_csv, write_json

from seintel.eval.metrics import bootstrap, brier, ece, roc_auc
from seintel.risk.features import build_split_table
from seintel.risk.model import make_variant

VARIANTS = ["static", "metrics", "full", "full_uncalibrated"]


def evaluate(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"auc": roc_auc(y, p), "brier": brier(y, p), "ece": ece(y, p),
            "brier_std": bootstrap(brier, y, p)[1], "auc_std": bootstrap(roc_auc, y, p)[1],
            "base_rate": float(y.mean()), "brier_base_rate": brier(y, np.full(len(y), y.mean())),
            "n": int(len(y)), "n_pos": int(y.sum())}


def llm_guess(spec, rows):
    from seintel.llm.base import make_llm
    llm = make_llm(spec)
    out = []
    for r in rows:
        prompt = (f"File {r['path']} in the Python project {r['repo']}: {r['sloc']} source lines, cyclomatic "
                  f"complexity sum {r['cc_sum']}, max {r['cc_max']}, maintainability index {r['mi']}. "
                  "Estimate the probability (0-1) that this file receives at least one bug-fix commit in the "
                  "project's next development period. Reply with only the number.")
        t = llm.complete("You estimate software defect risk.", [{"role": "user", "content": prompt}], 2000).text
        m = re.search(r"(0?\.\d+|1\.0+|[01])", t)
        out.append(float(m.group(1)) if m else 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    a = ap.parse_args()
    train = [x for r in REPOS for x in build_split_table(r, "train")]
    tune = [x for r in REPOS for x in build_split_table(r, "tune")]
    test = {r: build_split_table(r, "test") for r in REPOS}
    res, rows_out, models = {}, [], {}
    for v in VARIANTS:
        m = make_variant(v).fit(train, tune)
        models[v] = m
        res[v] = {}
        for r in REPOS:
            e = evaluate([x["y"] for x in test[r]], m.predict(test[r]))
            res[v][r] = e
            rows_out.append({"variant": v, "repo": r, **e})
        pooled = [x for r in REPOS for x in test[r]]
        res[v]["pooled"] = evaluate([x["y"] for x in pooled], m.predict(pooled))
        rows_out.append({"variant": v, "repo": "pooled", **res[v]["pooled"]})
    if a.llm:
        res["llm_only"] = {}
        for r in REPOS:
            e = evaluate([x["y"] for x in test[r]], llm_guess(a.llm, test[r]))
            res["llm_only"][r] = e
            rows_out.append({"variant": "llm_only", "repo": r, **e})
    wins = [r for r in REPOS if res["full"][r]["brier"] < res["metrics"][r]["brier"]]
    ece_ok = [r for r in REPOS if res["full"][r]["ece"] <= 0.10]
    res["hypothesis"] = {"criterion": "Brier(full) < Brier(metrics-only) on >= 3/4 repos and ECE(full) <= 0.10",
                         "brier_wins": wins, "ece_ok": ece_ok,
                         "supported": len(wins) >= 3 and len(ece_ok) == len(REPOS)}
    res["split_sizes"] = {"train": len(train), "tune": len(tune), "test": {r: len(test[r]) for r in REPOS},
                          "train_pos_rate": float(np.mean([x["y"] for x in train])),
                          "tune_pos_rate": float(np.mean([x["y"] for x in tune]))}
    models["full"].save(ROOT / "results/gap3/risk_model_full.json")
    res["full_model"] = models["full"].to_json()
    write_json("results/gap3/summary.json", res)
    write_csv("results/gap3/risk_test.csv", rows_out)
    # reliability diagram data for the paper figure
    pooled = [x for r in REPOS for x in test[r]]
    y = np.array([x["y"] for x in pooled])
    rel = []
    for v in ("full", "full_uncalibrated", "metrics"):
        p = models[v].predict(pooled)
        b = np.minimum((p * 10).astype(int), 9)
        for k in range(10):
            msk = b == k
            if msk.any():
                rel.append({"variant": v, "bin": k, "mean_pred": float(p[msk].mean()),
                            "frac_pos": float(y[msk].mean()), "n": int(msk.sum())})
    write_csv("results/gap3/reliability.csv", rel)
    for v in [*VARIANTS, *(["llm_only"] if a.llm else [])]:
        print(f"{v:18s} " + "  ".join(f"{r}: AUC={res[v][r]['auc']:.3f} B={res[v][r]['brier']:.4f} "
                                      f"ECE={res[v][r]['ece']:.3f}" for r in REPOS))
    print("hypothesis:", json.dumps(res["hypothesis"]))


if __name__ == "__main__":
    main()
