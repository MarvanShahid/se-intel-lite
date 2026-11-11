"""Generate the paper's figures and LaTeX tables from results/ only (no hand-copied numbers).

Outputs: results/figures/{gap2_rankings,gap3_reliability}.pdf/.png, paper/tables/*.tex
Usage: python scripts/make_paper_assets.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "results" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
REPOS = ["requests", "flask", "xarray", "pytest"]
# validated categorical palette, slots 1-3, in fixed order; hatches/markers make it grayscale-safe
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", f"results/figures/{name}.pdf")


def fig_gap2(g2):
    methods = ["loc", "cc", "graph_only", "hotspot", "churn", "churn_x_cc"]
    labels = ["LOC", "CC", "Graph", "Ours", "Churn", "Ch.×CC"]
    p10 = [g2[m]["mean_p@10"] for m in methods]
    nd = [g2[m]["mean_ndcg@10"] for m in methods]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = range(len(methods))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], p10, w, color=C1, label="P@10", edgecolor="white", linewidth=0.8)
    ax.bar([i + w / 2 + 0.01 for i in x], nd, w, color=C2, label="nDCG@10", hatch="////", edgecolor="white",
           linewidth=0.8)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean over 4 repositories")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    save(fig, "gap2_rankings")


def fig_gap3():
    rows = list(csv.DictReader(open(ROOT / "results/gap3/reliability.csv", encoding="utf-8")))
    spec = {"full": ("Full, calibrated (ours)", C1, "o", "-"),
            "full_uncalibrated": ("Full, uncalibrated", C2, "s", "--"),
            "metrics": ("Metrics only, calibrated", C3, "^", ":")}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), label="Perfect calibration")
    for v, (lab, col, mk, ls) in spec.items():
        r = [x for x in rows if x["variant"] == v]
        ax.plot([float(x["mean_pred"]) for x in r], [float(x["frac_pos"]) for x in r], color=col, marker=mk,
                markersize=4.5, linewidth=1.5, linestyle=ls, label=lab, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraction fixed")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "gap3_reliability")


def tex(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", f"paper/tables/{name}.tex")


def tab_gap2(g2):
    names = {"loc": "LOC (baseline)", "cc": "Complexity only", "graph_only": "Graph only",
             "hotspot": "Hotspot (ours)", "churn": "Churn", "churn_x_cc": "Churn $\\times$ CC"}
    lines = []
    for m, n in names.items():
        pr = g2[m]["per_repo"]
        lines.append(f"{n} & " + " & ".join(f"{pr[r]['p@10']:.1f}" for r in REPOS)
                     + f" & {g2[m]['mean_p@10']:.3f} & {g2[m]['mean_ndcg@10']:.3f} \\\\")
    tex("gap2", "\\begin{tabular}{lcccccc}\n\\toprule\nRanking & req. & flask & xarray & pytest & P@10 & nDCG@10 \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_gap3(g3):
    names = {"static": "Static only", "metrics": "Metrics only (static+change)", "full": "Full (ours)",
             "full_uncalibrated": "Full, no calibration"}
    lines = []
    for v, n in names.items():
        lines.append(f"{n} & " + " & ".join(f"{g3[v][r]['auc']:.2f} / {g3[v][r]['brier']:.3f} / {g3[v][r]['ece']:.2f}"
                                            for r in REPOS) + " \\\\")
    tex("gap3", "\\begin{tabular}{lcccc}\n\\toprule\nModel & requests & flask & xarray & pytest \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_data(g2):
    lines = []
    stats = json.load(open(ROOT / "results/graph_stats.json"))
    for r in REPOS:
        s, sp = stats[r], g2["splits"][r]
        lines.append(f"{r} & {s['nodes'].get('file', 0)} & {s['nodes'].get('function', 0)} & "
                     f"{sum(s['edges'].values())} & {sp['n_commits']} & {sp['n_bugfix_commits']} & "
                     f"{sp['test']['snapshot_date'][:4]}--{sp['test']['window_end_date'][:4]} \\\\")
    tex("data", "\\begin{tabular}{lrrrrrc}\n\\toprule\nRepo & Files & Funcs & Edges & Commits & Fix commits & "
        "Test window \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_gap_resolution():
    md = (ROOT / "results/gap_resolution.md").read_text(encoding="utf-8").strip().splitlines()
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in md[2:]]
    esc = lambda s: s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("×", "$\\times$") \
        .replace("−", "-").replace("≥", "$\\geq$").replace("≤", "$\\leq$")
    body = "\n".join(" & ".join(esc(c) for c in r) + " \\\\ \\midrule" for r in rows)
    tex("gap_resolution", "\\begin{tabular}{p{1.6cm}p{2.6cm}p{2.0cm}p{2.8cm}p{3.2cm}p{1.2cm}}\n\\toprule\n"
        "Gap & Method / component & Metric & Base paper result & Our result & Addressed? \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    g2 = json.load(open(ROOT / "results/gap2/summary.json"))
    g3 = json.load(open(ROOT / "results/gap3/summary.json"))
    fig_gap2(g2)
    fig_gap3()
    tab_gap2(g2)
    tab_gap3(g3)
    tab_data(g2)
    tab_gap_resolution()
