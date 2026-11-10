"""E1 (B0, B1), E2 (B2 = LocAgent-style base paper approach), E3 (P) and E5 (ablations of P) on SWE-QA.

Each configuration is run `--runs` times (default 3) on the same questions with the same LLM and one fixed judge.
Results: results/qa/<split>/<system>/run<r>.jsonl; trajectories: logs/<split>/<system>/run<r>/<qid>.json

Examples
  python experiments/run_qa.py --systems B0 B1 B2 P                     # E1-E3 on the 172 test questions
  python experiments/run_qa.py --systems A1_no_traverse A2_hop1 A3_no_quality A3b_no_risk \
                                         A7_no_qa_prompt A8_no_verifier A6_small_llm   # E5
  python experiments/run_qa.py --systems P --split dev --runs 1         # development subset (20 questions)
  python experiments/run_qa.py --systems B2 P --llm mock --no-judge --limit 2 --runs 1  # offline smoke run
"""
import argparse

from _common import ROOT

from seintel.agent.factory import SYSTEMS
from seintel.eval.runner import run_system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["B0", "B1", "B2", "P"], choices=list(SYSTEMS))
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--llm", default=None, help="e.g. anthropic:claude-opus-5-5 (default: $SEINTEL_LLM)")
    ap.add_argument("--judge", default=None, help="fixed judge model (default: $SEINTEL_JUDGE)")
    ap.add_argument("--repos", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first n questions per repo (smoke runs only)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default=None, help="output root (default: repository root)")
    a = ap.parse_args()
    out = ROOT / a.out if a.out else None
    for s in a.systems:
        run_system(s, a.split, a.runs, a.llm, a.judge if not a.llm or a.llm != "mock" else "mock", a.repos,
                   a.limit, out, judge=not a.no_judge)


if __name__ == "__main__":
    main()
