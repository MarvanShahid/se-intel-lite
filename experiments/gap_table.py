"""Build the gap-resolution table (README + paper) from the saved results files only.

Writes results/gap_resolution.md. Cells whose experiment has not been run say "not run".
"Addressed?" is filled from the pre-registered Phase 1 criteria: Fully = criterion met; Partially = direction
right but criterion not met; Not = no improvement.

Usage: python experiments/gap_table.py [--split test]
"""
import argparse
import csv
import json

from _common import ROOT


def load(p):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else None


def verdict(supported, improved):
    return "Fully" if supported else ("Partially" if improved else "Not")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = []
    # Gap 1
    g1 = load(f"results/e4_gap1_{a.split}.json")
    qa = {}
    f = ROOT / f"results/qa_summary_{a.split}.csv"
    if f.exists():
        qa = {r["system"]: r for r in csv.DictReader(open(f, encoding="utf-8"))}
    if g1 and "P_vs_B1" in g1 and "P" in qa:
        g = g1["P_vs_B1"]
        rows.append(["Gap 1: repository QA with typed multi-hop graph agent",
                     "ReAct graph agent + QA protocol + citation verifier (P)",
                     "SWE-QA overall /100 (multi-hop)",
                     f"B2 (LocAgent-style, reproduced): {float(qa['B2']['score_mean']):.1f} "
                     f"({float(qa['B2']['multihop_mean']):.1f})" if "B2" in qa else "B2 not run",
                     f"P: {float(qa['P']['score_mean']):.1f}+-{float(qa['P']['score_std']):.1f} "
                     f"({float(qa['P']['multihop_mean']):.1f}); vs B1 {g['overall_rel_gain_pct']:+.1f}% "
                     f"(multi-hop {g['multihop_rel_gain_pct']:+.1f}%), {g['token_ratio']:.2f}x tokens",
                     verdict(g1["hypothesis"]["supported"], g["overall_rel_gain_pct"] > 0)])
    else:
        rows.append(["Gap 1: repository QA with typed multi-hop graph agent",
                     "ReAct graph agent + QA protocol + citation verifier (P)", "SWE-QA overall /100",
                     "not run", "not run (needs LLM API key; see README)", "-"])
    # Gap 2
    g2 = load("results/gap2/summary.json")
    if g2:
        rows.append(["Gap 2: quality signals tied to the dependency graph",
                     f"RunQuality/GetHotspots + hotspot = z(CC) + {g2['weights']['w_in']}z(fan-in) + "
                     f"{g2['weights']['w_out']}z(fan-out)",
                     "P@10 / nDCG@10 vs later bug-fix files (test window)",
                     "LocAgent: no quality signal. LOC ranking: "
                     f"{g2['loc']['mean_p@10']:.3f} / {g2['loc']['mean_ndcg@10']:.3f}",
                     f"{g2['hotspot']['mean_p@10']:.3f} / {g2['hotspot']['mean_ndcg@10']:.3f} "
                     f"({g2['hypothesis']['delta_pp']:+.1f} pp P@10 vs LOC)",
                     verdict(g2["hypothesis"]["supported"], g2["hypothesis"]["delta_pp"] > 0)])
    # Gap 3
    g3 = load("results/gap3/summary.json")
    if g3:
        repos = [r for r in g3["full"] if r != "pooled"]
        fb = sum(g3["full"][r]["brier"] for r in repos) / len(repos)
        mb = sum(g3["metrics"][r]["brier"] for r in repos) / len(repos)
        fe = sum(g3["full"][r]["ece"] for r in repos) / len(repos)
        fa = sum(g3["full"][r]["auc"] for r in repos) / len(repos)
        ma = sum(g3["metrics"][r]["auc"] for r in repos) / len(repos)
        rows.append(["Gap 3: calibrated, graph-aware risk under a time-based split",
                     "GetRisk: logistic regression (static + change + graph) + Platt calibration",
                     "Brier / ECE / ROC AUC on later period (mean of 4 repos)",
                     f"LocAgent: no risk estimate. Metrics-only model: Brier {mb:.4f}, AUC {ma:.3f}",
                     f"Brier {fb:.4f}, ECE {fe:.3f}, AUC {fa:.3f}; Brier better on "
                     f"{len(g3['hypothesis']['brier_wins'])}/4 repos",
                     verdict(g3["hypothesis"]["supported"], len(g3["hypothesis"]["brier_wins"]) >= 2)])
    hdr = ["Gap", "Method / component", "Metric", "Base paper result", "Our result", "Addressed?"]
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
    md += "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"
    (ROOT / "results/gap_resolution.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
