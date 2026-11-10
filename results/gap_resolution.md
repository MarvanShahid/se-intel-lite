| Gap | Method / component | Metric | Base paper result | Our result | Addressed? |
|---|---|---|---|---|---|
| Gap 1: repository QA with typed multi-hop graph agent | ReAct graph agent + QA protocol + citation verifier (P) | SWE-QA overall /100 | not run | not run (needs LLM API key; see README) | - |
| Gap 2: quality signals tied to the dependency graph | RunQuality/GetHotspots + hotspot = z(CC) + 0.25z(fan-in) + 0.25z(fan-out) | P@10 / nDCG@10 vs later bug-fix files (test window) | LocAgent: no quality signal. LOC ranking: 0.875 / 0.851 | 0.850 / 0.862 (-2.5 pp P@10 vs LOC) | Not |
| Gap 3: calibrated, graph-aware risk under a time-based split | GetRisk: logistic regression (static + change + graph) + Platt calibration | Brier / ECE / ROC AUC on later period (mean of 4 repos) | LocAgent: no risk estimate. Metrics-only model: Brier 0.2043, AUC 0.762 | Brier 0.2072, ECE 0.218, AUC 0.744; Brier better on 2/4 repos | Partially |
