"""E4 (Gap 2) + its ablation: graph-aware hotspot ranking vs LOC and complexity-only rankings.

Hypothesis (Phase 1 B.10): Precision@10 of the hotspot ranking is >= 10 percentage points above the LOC ranking,
averaged over the four test repositories.

Protocol: weights tuned on the TUNE split only; TEST split evaluated once. Deterministic (no LLM), so a single run;
uncertainty is reported with a bootstrap over files.

Usage: python experiments/gap2_hotspots.py
"""
import numpy as np
from _common import REPOS, write_csv, write_json

from seintel.eval.metrics import ndcg_at_k, precision_at_k
from seintel.quality import hotspot
from seintel.risk.features import build_split_table, split_info

METHODS = ["loc", "cc", "graph_only", "hotspot", "churn", "churn_x_cc"]
PRIMARY = {"loc": "B-LOC (baseline)", "cc": "B-CC (complexity only = hotspot without graph)",
           "graph_only": "graph only (fan-in + fan-out)", "hotspot": "P hotspot (complexity + graph)",
           "churn": "reference: churn only", "churn_x_cc": "reference: churn x complexity"}


def bootstrap_p10(rows, method, w, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        # duplicate paths would collide; keep first occurrence
        seen, uniq = set(), []
        for r in sample:
            if r["path"] not in seen:
                seen.add(r["path"]); uniq.append(r)
        vals.append(hotspot.evaluate(uniq, method, w)["p@10"])
    return float(np.std(vals))


def main():
    tune = {r: build_split_table(r, "tune") for r in REPOS}
    test = {r: build_split_table(r, "test") for r in REPOS}
    w, trials = hotspot.tune_weights(tune, "ndcg@10")
    write_json("results/gap2/weights.json", {"w_in": w[0], "w_out": w[1], "tuned_on": "tune split, mean nDCG@10",
                                             "grid": list(hotspot.WEIGHT_GRID)})
    write_json("results/gap2/tuning_trials.json", trials)
    rows_out, summary = [], {}
    for m in METHODS:
        per = {}
        for r in REPOS:
            e = hotspot.evaluate(test[r], m, w)
            e["p@10_boot_std"] = bootstrap_p10(test[r], m, w)
            e["n_files"], e["n_fix_files"] = len(test[r]), sum(x["y"] for x in test[r])
            per[r] = e
            rows_out.append({"method": m, "repo": r, **e})
        summary[m] = {"label": PRIMARY[m], "per_repo": per,
                      "mean_p@10": float(np.mean([per[r]["p@10"] for r in REPOS])),
                      "mean_ndcg@10": float(np.mean([per[r]["ndcg@10"] for r in REPOS]))}
    delta = (summary["hotspot"]["mean_p@10"] - summary["loc"]["mean_p@10"]) * 100
    summary["hypothesis"] = {"criterion": "mean P@10(hotspot) - mean P@10(LOC) >= 10 percentage points",
                             "delta_pp": delta, "supported": delta >= 10}
    summary["weights"] = {"w_in": w[0], "w_out": w[1]}
    summary["splits"] = {r: split_info(r) for r in REPOS}
    # top-10 lists for the human usefulness rating (30 explanations, rated by two members)
    expl = []
    for r in REPOS:
        ranked = hotspot.rank(test[r], "hotspot", w)
        by = {x["path"]: x for x in test[r]}
        for i, p in enumerate(ranked[:10]):
            expl.append({"repo": r, "rank": i + 1, "path": p, "explanation": hotspot.explain(by[p], i + 1, len(ranked)),
                         "later_fix_commits": by[p]["n_fixes"], "rater1_1to5": "", "rater2_1to5": ""})
    write_csv("results/gap2/hotspot_test.csv", rows_out)
    write_json("results/gap2/summary.json", summary)
    write_csv("results/gap2/explanations_to_rate.csv", expl[:30])
    print(f"\nweights={w}")
    for m in METHODS:
        print(f"{m:12s} P@10={summary[m]['mean_p@10']:.3f} nDCG@10={summary[m]['mean_ndcg@10']:.3f}  "
              + "  ".join(f"{r}={summary[m]['per_repo'][r]['p@10']:.1f}" for r in REPOS))
    print("hypothesis:", summary["hypothesis"])


if __name__ == "__main__":
    main()
