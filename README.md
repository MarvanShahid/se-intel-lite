# SE-INTEL Lite (G3-13): a graph-grounded repository agent

SE-INTEL Lite answers developer questions about a Python repository with a ReAct agent that explores a code
graph through typed tools (LocAgent [1] style). It also reports **quality hotspots** tied to the dependency graph
and a **calibrated defect-risk probability** for source files.

Group: Abdullah Bin Nasir (23I-0803), Marvan Shahid (23I-0800). Base paper: LocAgent, ACL 2025.

| Gap (Phase 1) | Component in the system | Switch for ablation |
|---|---|---|
| 1. Graph agents are not tested on repository QA | ReAct agent with typed multi-hop `traverse_graph`, QA protocol prompt, citation verifier | `A1_no_traverse`, `A2_hop1`, `A7_no_qa_prompt`, `A8_no_verifier` |
| 2. Quality signals are not tied to the dependency graph | `run_quality`, `get_hotspots`, hotspot score = complexity + fan-in + fan-out | `A3_no_quality`; ranking `cc` (no graph) |
| 3. Risk estimates are idealised, not graph-aware, not calibrated | `get_risk`: logistic regression (static + change + graph), Platt-calibrated, time-based split | `A3b_no_risk`; model `metrics` (no graph), `full_uncalibrated` |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the diagram and component table.

```
repo@pinned commit ──AST──> code graph ──> entity index / chunk index / file dependency graph
        └──git log──> change features + bug-fix labels (time-based train/tune/test)
                                   │                         │
                         hotspot ranker (G2)        calibrated risk model (G3)
                                   │                         │
   Claude ⇄ ReAct loop (budget 12) ⇄ search_entity · traverse_graph · retrieve_entity · run_quality · get_hotspots · get_risk
                 └── citation verifier (G1) ──> answer + evidence trail + logged trajectory
```

## Install

```sh
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sh scripts/setup_data.sh                             # clones the 4 repos at pinned commits + SWE-QA-Bench
cp .env.example .env                                 # then set ANTHROPIC_API_KEY (or run `ant auth login`)
```

Python 3.10 or newer is required. The results in `results/` were produced with Python 3.14.3 on Windows 11.

### Configuration and environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Claude API key. Only needed for Gap 1 QA runs, `ask` and the LLM-only risk baseline |
| `SEINTEL_LLM` | `anthropic:claude-opus-5-5` | agent/baseline model (`provider:model`; `mock` for offline runs) |
| `SEINTEL_JUDGE` | `anthropic:claude-opus-5-5` | one fixed judge model for every system |
| `SEINTEL_EFFORT` | `medium` | thinking effort of the agent model |

Repositories and commits are frozen in [configs/repos.yaml](configs/repos.yaml). The SWE-QA-Bench data is pinned
to commit `c13deac7`: 48 questions per repository (192 in total). The split is fixed (seed 13): 5 questions per
repository form the **dev** set (20) and the other **172 are the test set**. The dev set is used for design
decisions; the test set is not.

## Run the system

```sh
export PYTHONPATH=src                                   # or: pip install -e .
python -m seintel ask --repo flask "How does Flask dispatch a request to a view function?" --trace
python -m seintel ask --repo /path/to/any/python/repo --system B2 "Where is the config parsed?"
python -m seintel hotspots --repo flask --top 10
python -m seintel risk --repo flask src/flask/app.py
python -m seintel graph --repo requests
```

## Reproduce the experiments

All scripts write to `results/` and the agent trajectories go to `logs/`. `sh experiments/run_all.sh` runs
everything, and `sh experiments/run_all.sh --offline` runs only the parts that need no API calls.

| Exp. | What | Command | Output |
|---|---|---|---|
| E1 | B0 plain LLM, B1 flat RAG | `python experiments/run_qa.py --systems B0 B1` | `results/qa/test/B0,B1/run{0,1,2}.jsonl` |
| E2 | B2 LocAgent-style agent (reproduced on our data) | `python experiments/run_qa.py --systems B2` | `results/qa/test/B2/` |
| E3 | P, the full system | `python experiments/run_qa.py --systems P` | `results/qa/test/P/` |
| E4 Gap 1 | P vs B1/B2 overall + multi-hop, Wilcoxon, token ratio | `python experiments/analyze_qa.py` | `results/e4_gap1_test.json` |
| E4 Gap 2 | hotspot ranking vs LOC / CC | `python experiments/gap2_hotspots.py` | `results/gap2/` |
| E4 Gap 3 | calibrated risk vs metrics-only (+ `--llm` LLM-only guess) | `python experiments/gap3_risk.py` | `results/gap3/` |
| E5 | ablations A1–A8 | `python experiments/run_qa.py --systems A1_no_traverse A2_hop1 A3_no_quality A3b_no_risk A7_no_qa_prompt A8_no_verifier A6_small_llm`, then `analyze_qa.py` | `results/e5_ablation_test.csv`; A4/A5 in `results/gap3/` |
| E6 | tokens, cost, latency, tool calls, reliability | `python experiments/analyze_qa.py` | `results/e6_efficiency_test.csv` |
| Failure analysis | categories + representative trajectories | `python experiments/analyze_qa.py` | `results/failures/categories_test.json` |
| Gap table | from the saved results only | `python experiments/gap_table.py` | `results/gap_resolution.md` |
| Label noise | 50 + 50 commits to label by hand | `python scripts/sample_fix_commits.py [--score]` | `results/label_noise/` |

Protocol: every QA configuration runs 3 times on the same 172 test questions with the same model, and we report
the mean and standard deviation over runs. Current Claude models do not accept `temperature`, so runs are
stochastic. Every LLM call is cached on disk under `.cache/llm`, keyed by run id, so re-running a script
reproduces the saved numbers without new API calls. A smoke run with no API key:
`python experiments/run_qa.py --systems B2 P --llm mock --no-judge --split dev --runs 1 --limit 1 --out tmp_smoke`.

Tests: `python -m pytest -q` (29 tests: parser and graph, indexes, tools, hop limit, agent loop, budget,
verifier, history mining, labels, snapshot extraction, metrics, risk model, hotspot tuning, judge parsing).

## Gap-resolution table

Generated by `experiments/gap_table.py` from `results/`. Gap 1 is empty until the QA experiments are run with an
API key.

| Gap | Method / component | Metric | Base paper result | Our result | Addressed? |
|---|---|---|---|---|---|
| Gap 1 | ReAct graph agent + QA protocol + citation verifier (P) | SWE-QA overall /100 (multi-hop) | not run yet | not run yet | – |
| Gap 2 | hotspot = z(CC) + 0.25 z(fan-in) + 0.25 z(fan-out) (weights tuned on the tune split) | mean P@10 / nDCG@10 against files fixed in the test window | LocAgent gives no quality signal. LOC ranking: 0.875 / 0.851 | 0.850 / 0.862 (−2.5 pp P@10 vs LOC) | Not |
| Gap 3 | logistic regression (static + change + graph) + Platt calibration | Brier / ECE / AUC, mean of 4 repos, later period | LocAgent gives no risk estimate. Metrics-only model: Brier 0.204, AUC 0.762 | Brier 0.207, ECE 0.218, AUC 0.744; lower Brier than metrics-only on 2/4 repos; calibration lowers Brier on 4/4 | Partially |

"Base paper result" for Gaps 2 and 3: LocAgent has no quality or risk output, so the comparison point is the
simpler baseline named in Phase 1 (computed by us on the same data).

### What the Gap 2/3 numbers mean (honest reading)

- **The Phase 1 hypotheses for Gap 2 and Gap 3 are not supported on the test split.** The tuning procedure chose
  small graph weights (0.25, 0.25). On the test split the graph-aware ranking ties complexity-only ranking on
  P@10 and is slightly better on nDCG@10 (0.862 vs 0.848). It is 2.5 points below LOC on P@10.
- **The evaluation is close to saturated.** The test window covers the last 25% of each history (2016–2025 for
  requests, 2020–2025 for pytest), so 60–72% of source files receive at least one bug fix. xarray and pytest reach
  P@10 = 1.0 under every ranking. requests (15 source files) and flask (20) are too small for P@10 to separate
  methods: there the bootstrap std of P@10 is about 0.1.
- **Calibration helps; graph features do not.** Platt scaling on the tune split lowers Brier and ECE on all four
  repositories, but ECE stays above the 0.10 target (0.16–0.30), because the base rate shifts between windows.
  Graph features improve Brier on requests and flask only. The full model's per-file contributions show
  collinearity (for example, `fan_in` negative while `dependents_2hop` is positive), so explanations must be read
  with care.
- A shorter, fixed-length label window (for example, 6 months) and function-level units would give a less
  saturated test. We did **not** change the design after seeing test results; this is listed as future work.

## Changes since Phase 1

| Change | Reason |
|---|---|
| Implemented the modules that were missing from the Phase 1 snapshot (`llm`, `index`, `eval`, baselines) | The baseline code imported them, but they were not in the repository |
| B1 uses a local BM25 + TF-IDF hybrid (rank fusion) instead of Chroma dense embeddings | No embedding API or extra service needed; fully reproducible offline |
| CLI instead of the Next.js web UI | The fallback named in the Phase 1 risk table ("drop the web interface first") |
| git CLI instead of PyDriller for history mining | Same information, much faster on pytest (12k commits) |
| Time split defined on commit-count fractions: train = snapshot 40% → window 40–60%, tune = 60% → 75%, test = 75% → 100% | Commit fractions avoid empty windows in quiet periods; labels for training stay inside the first 60% |
| Risk model trained pooled over the 4 repos, evaluated per repo | Per-repo training has too few files (15–88) |
| Vendored code (`*/packages/`, `vendor/`, …) excluded from the hotspot/risk population | Found after the first Gap 2 run: 78 of requests' 93 "source" files were vendored urllib3/chardet. Results before the fix are kept in `results/gap2/pre_vendor_fix/` (see devlog) |
| Model: Claude Opus 5.5 for agent and judge, Claude Haiku 5.5 for A6; no temperature; 3 runs | Current models reject sampling parameters; repeated runs give mean ± std |
| Judge: the official SWE-QA rubric (5 dimensions × 20 = 100), copied verbatim | Comparability with SWE-QA |
| Multi-hop label: the reference answer names entities from ≥ 2 files | SWE-QA has no hop annotation; this label is deterministic and reproducible |
| Added ablations A7 (no QA prompt), A8 (no verifier) and A3b (no risk tool) | Each Gap 1 sub-component must be switchable |

## Repository layout

```
src/seintel/   agent/ (ReAct loop, state, system factory)  tools/  graph/  index/  llm/  eval/
               history/ (git mining)  quality/ (metrics, hotspots)  risk/ (features, model)  baselines/  cli.py
experiments/   run_qa.py analyze_qa.py gap2_hotspots.py gap3_risk.py gap_table.py run_all.sh
results/       saved results used in the paper      logs/  agent trajectories
tests/         deterministic components             docs/  architecture.md devlog.md
configs/       frozen repositories                   scripts/ setup_data.sh graph_stats.py sample_fix_commits.py
```

## Status and remaining work for the group

- [ ] Run E1–E3, E5 and E6 with an API key (start with `--split dev` to measure cost per question), then run
      `analyze_qa.py` and `gap_table.py`. Add a representative sample of `logs/` to the repository.
- [ ] Label `results/label_noise/sample.csv` (two members), then run `sample_fix_commits.py --score`.
- [ ] Rate the 30 hotspot explanations in `results/gap2/explanations_to_rate.csv` (two raters, 1–5).
- [ ] Check judge agreement by hand on 30 answers.
- [ ] Commit history from both members, plus the `phase2-final` tag/release.
- [ ] Write the paper yourselves. The course forbids AI-generated writing; every number must come from `results/`.

## References

[1] Z. Chen et al., "LocAgent: Graph-guided LLM agents for code localization," ACL 2025,
https://doi.org/10.18653/v1/2025.acl-long.426 · SWE-QA: W. Peng et al., https://arxiv.org/abs/2509.14635
