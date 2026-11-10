#!/usr/bin/env sh
# Reproduce every experiment (E1-E6) and the gap-resolution table. Results go to results/, traces to logs/.
#   sh experiments/run_all.sh            full run (needs ANTHROPIC_API_KEY or `ant auth login`)
#   sh experiments/run_all.sh --offline  only the deterministic parts (Gap 2, Gap 3, tests) - no API calls
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY -m pytest -q
$PY scripts/graph_stats.py
$PY experiments/gap2_hotspots.py          # E4 Gap 2 (+ hotspot-without-graph ablation)
$PY experiments/gap3_risk.py              # E4 Gap 3 + E5 A4 (no graph) / A5 (no calibration)

if [ "$1" != "--offline" ]; then
  $PY experiments/run_qa.py --systems P --split dev --runs 1                    # dev subset (design decisions only)
  $PY experiments/run_qa.py --systems B0 B1 B2 P --split test --runs 3          # E1, E2, E3
  $PY experiments/run_qa.py --systems A1_no_traverse A2_hop1 A3_no_quality A3b_no_risk \
                                      A7_no_qa_prompt A8_no_verifier A6_small_llm --split test --runs 3   # E5
  $PY experiments/gap3_risk.py --llm "${SEINTEL_LLM:-anthropic:claude-opus-5-5}"  # Gap 3 LLM-only risk guess
  $PY experiments/analyze_qa.py --split test                                     # E4 Gap 1, E5, E6, failures
fi
$PY experiments/gap_table.py --split test
