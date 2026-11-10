"""Write a fixed random sample of 50 keyword-labelled bug-fix commits (and 50 non-fix commits) for the manual
label-noise check (Phase 1 B.7). Two members fill the `is_real_bugfix` column; then run with --score.

Usage: python scripts/sample_fix_commits.py            -> results/label_noise/sample.csv
       python scripts/sample_fix_commits.py --score    -> results/label_noise/noise.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seintel.history.mining import is_source_file, load_history  # noqa: E402

OUT = ROOT / "results" / "label_noise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "sample.csv"
    if a.score:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        lab = [r for r in rows if r["is_real_bugfix"].strip() in ("0", "1")]
        fp = [r for r in lab if r["keyword_label"] == "1"]
        fn = [r for r in lab if r["keyword_label"] == "0"]
        res = {"n_labelled": len(lab),
               "precision_of_fix_label": sum(r["is_real_bugfix"] == "1" for r in fp) / max(1, len(fp)),
               "miss_rate_of_nonfix_label": sum(r["is_real_bugfix"] == "1" for r in fn) / max(1, len(fn))}
        (OUT / "noise.json").write_text(json.dumps(res, indent=2))
        print(res)
        return
    rnd = random.Random(2026)
    rows = []
    per_repo = {"requests": 13, "flask": 13, "xarray": 12, "pytest": 12}   # 50 per label
    for repo, n in per_repo.items():
        cs = [c for c in load_history(str(ROOT / "repos" / repo)) if any(is_source_file(p) for p in c.files)]
        for label, pool in ((1, [c for c in cs if c.bugfix]), (0, [c for c in cs if not c.bugfix])):
            for c in rnd.sample(pool, n):
                rows.append({"repo": repo, "sha": c.sha[:12], "keyword_label": label,
                             "subject": c.message.splitlines()[0][:120],
                             "files": ";".join(p for p in c.files if is_source_file(p))[:200], "is_real_bugfix": ""})
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {f} ({sum(r['keyword_label'] for r in rows)} fix-labelled, "
          f"{sum(1 - r['keyword_label'] for r in rows)} non-fix)")


if __name__ == "__main__":
    main()
