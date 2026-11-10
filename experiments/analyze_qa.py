"""E1-E6 analysis of saved QA results (no LLM calls).

Writes
  results/qa_summary.csv            per system: mean +- std over runs of score, multi-hop score, tokens, cost ...
  results/e4_gap1.json              Gap 1 hypothesis test: P vs B1 overall / multi-hop, token ratio, Wilcoxon
  results/e5_ablation.csv           delta of each ablation vs P
  results/e6_efficiency.csv         cost, latency, tool calls and reliability per system
  results/failures/categories.json  failure categories of P with frequencies + representative cases

Usage: python experiments/analyze_qa.py [--split test] [--root .]
"""
import argparse
import json
from collections import Counter, defaultdict
from statistics import mean

import _common
from _common import ROOT, write_csv, write_json

from seintel.agent.factory import SYSTEMS
from seintel.eval.metrics import mean_std, wilcoxon
from seintel.eval.runner import load_results

FAIL_THRESHOLD = 50   # overall judge score < 50/100 counts as a failure (fixed on the dev split)


def per_run(rows, key, filt=lambda r: True):
    by = defaultdict(list)
    for r in rows:
        if filt(r):
            v = r.get(key) if key != "score" else (r.get("scores") or {}).get("overall")
            if v is not None:
                by[r["run"]].append(float(v))
    return [mean(v) for v in by.values() if v]


def per_question(rows):
    """qid -> mean overall score across runs."""
    by = defaultdict(list)
    for r in rows:
        s = (r.get("scores") or {}).get("overall")
        if s is not None:
            by[r["qid"]].append(s)
    return {q: mean(v) for q, v in by.items()}


def categorize(r) -> str:
    """Rule-based failure categories (assigned automatically, then checked by hand on a sample)."""
    if r.get("error"):
        return "system_error"
    if r.get("llm_errors"):
        return "llm_error"
    if r.get("budget_exhausted"):
        return "budget_exhausted"
    if r.get("invalid_json", 0) >= 2:
        return "protocol_violation"
    if r.get("ungrounded", 0) > 0:
        return "ungrounded_citation"
    rec = r.get("evidence_recall")
    if rec is not None and rec == 0:
        return "retrieval_miss"          # never reached any file the reference relies on
    if rec is not None and rec < 0.5:
        return "partial_retrieval"
    return "reasoning_or_synthesis"      # found the right code, answer still judged weak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = ROOT / a.root if a.root else ROOT
    _common.OUT_ROOT = root            # analysis files go next to the results they summarise
    data ={s: load_results(s, a.split, root) for s in SYSTEMS}
    data = {s: v for s, v in data.items() if v}
    if not data:
        print("no QA results found; run experiments/run_qa.py first")
        return
    summary = []
    for s, rows in data.items():
        def ms(key, filt=lambda r: True):
            return mean_std(per_run(rows, key, filt))
        sc, mh, sh = ms("score"), ms("score", lambda r: r["multi_hop"]), ms("score", lambda r: not r["multi_hop"])
        row = {"system": s, "llm": rows[0].get("llm"), "runs": len({r["run"] for r in rows}),
               "n_questions": len({r["qid"] for r in rows}),
               "score_mean": sc[0], "score_std": sc[1], "multihop_mean": mh[0], "multihop_std": mh[1],
               "singlehop_mean": sh[0], "singlehop_std": sh[1]}
        for k in ("total_tokens", "cost_usd", "latency_s", "tool_calls", "evidence_recall"):
            m_, s_ = ms(k)
            row[f"{k}_mean"], row[f"{k}_std"] = m_, s_
        for k in ("budget_exhausted", "revisions", "ungrounded", "invalid_json", "llm_errors"):
            row[f"{k}_rate"] = mean(1.0 if r.get(k) else 0.0 for r in rows)
        row["error_rate"] = mean(1.0 if r.get("error") else 0.0 for r in rows)
        for qt in ("what", "why", "where", "how"):
            row[f"score_{qt}"] = ms("score", lambda r, qt=qt: r["qtype"] == qt)[0]
        summary.append(row)
    write_csv(f"results/qa_summary_{a.split}.csv", summary)
    by = {r["system"]: r for r in summary}

    # ---- E4 Gap 1: P vs B1 (and vs B2, the base paper approach)
    gap1 = {}
    for base in ("B1", "B2", "B0"):
        if "P" in data and base in data:
            qp, qb = per_question(data["P"]), per_question(data[base])
            common = sorted(set(qp) & set(qb))
            mh = {r["qid"] for r in data["P"] if r["multi_hop"]}
            pm, bm = by["P"]["score_mean"], by[base]["score_mean"]
            pmh, bmh = by["P"]["multihop_mean"], by[base]["multihop_mean"]
            gap1[f"P_vs_{base}"] = {
                "overall_rel_gain_pct": (pm - bm) / bm * 100 if bm else None,
                "multihop_rel_gain_pct": (pmh - bmh) / bmh * 100 if bmh else None,
                "token_ratio": by["P"]["total_tokens_mean"] / by[base]["total_tokens_mean"]
                if by[base]["total_tokens_mean"] else None,
                "wilcoxon_overall": wilcoxon([qp[q] for q in common], [qb[q] for q in common]),
                "wilcoxon_multihop": wilcoxon([qp[q] for q in common if q in mh], [qb[q] for q in common if q in mh]),
            }
    if "P_vs_B1" in gap1:
        g = gap1["P_vs_B1"]
        gap1["hypothesis"] = {"criterion": ">= +5% relative overall and >= +10% relative on multi-hop vs B1, "
                                           "<= 2x tokens",
                              "supported": bool(g["overall_rel_gain_pct"] is not None and g["overall_rel_gain_pct"] >= 5
                                                and (g["multihop_rel_gain_pct"] or 0) >= 10
                                                and (g["token_ratio"] or 99) <= 2)}
    write_json(f"results/e4_gap1_{a.split}.json", gap1)

    # ---- E5 ablations
    if "P" in by:
        abl = [{"system": s, "score": by[s]["score_mean"], "score_std": by[s]["score_std"],
                "delta_vs_P": by[s]["score_mean"] - by["P"]["score_mean"],
                "multihop_delta_vs_P": by[s]["multihop_mean"] - by["P"]["multihop_mean"],
                "tokens_ratio_vs_P": by[s]["total_tokens_mean"] / by["P"]["total_tokens_mean"]
                if by["P"]["total_tokens_mean"] else None}
               for s in by if s.startswith("A")]
        write_csv(f"results/e5_ablation_{a.split}.csv", abl)

    # ---- E6 efficiency & reliability
    keys = ["system", "llm", "total_tokens_mean", "total_tokens_std", "cost_usd_mean", "cost_usd_std", "latency_s_mean",
            "latency_s_std", "tool_calls_mean", "budget_exhausted_rate", "revisions_rate", "ungrounded_rate",
            "invalid_json_rate", "llm_errors_rate", "error_rate"]
    write_csv(f"results/e6_efficiency_{a.split}.csv", [{k: r.get(k) for k in keys} for r in summary])

    # ---- failure analysis for P
    if "P" in data:
        fails = [r for r in data["P"] if ((r.get("scores") or {}).get("overall") or 0) < FAIL_THRESHOLD]
        cats = Counter(categorize(r) for r in fails)
        ok = sorted([r for r in data["P"] if ((r.get("scores") or {}).get("overall") or 0) >= FAIL_THRESHOLD],
                    key=lambda r: -(r["scores"]["overall"]))
        rep = {"threshold": FAIL_THRESHOLD, "n_answers": len(data["P"]), "n_failures": len(fails),
               "categories": {k: {"count": v, "share": v / len(fails)} for k, v in cats.most_common()},
               "success_examples": [f"logs/{a.split}/P/run{r['run']}/{r['qid']}.json" for r in ok[:2]],
               "failure_examples": {c: [f"logs/{a.split}/P/run{r['run']}/{r['qid']}.json"
                                        for r in fails if categorize(r) == c][:2] for c in cats}}
        write_json(f"results/failures/categories_{a.split}.json", rep)
    for r in summary:
        print(f"{r['system']:16s} score={r['score_mean']:.1f}+-{r['score_std']:.1f} "
              f"multihop={r['multihop_mean']:.1f} tok={r['total_tokens_mean']:.0f} cost=${r['cost_usd_mean']:.4f}")
    print(json.dumps(gap1.get("hypothesis", {}), indent=1))


if __name__ == "__main__":
    main()
