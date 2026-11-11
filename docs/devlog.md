# Development log

Each entry: date, the failure or limitation we observed, the change, and its measured effect.
Only changes that were actually made and measured are listed. Add new entries as the QA experiments run.

| # | Date | Observed failure / limitation | Change | Measured effect |
|---|---|---|---|---|
| 1 | 2026-11-10 | The Phase 1 code imported `seintel.llm`, `seintel.index` and `seintel.eval`, but those modules were not in the repository, so nothing ran | Implemented the LLM layer (Claude, disk cache, mock), the layered entity index, the chunk index, the dataset/split loader, the judge and the metrics | The pipeline runs end to end. 29 tests pass. Offline mock smoke run on 4 dev questions × 4 systems: 0 errors |
| 2 | 2026-11-10 | The agent used `max_tokens=900` and `temperature=0`. Current Claude models think adaptively (thinking counts toward `max_tokens`) and reject `temperature` | `max_tokens=16000`, removed sampling parameters, 3 repeated runs per configuration, disk cache keyed by run id | No 400 errors from sampling parameters. Run-to-run variance is now measured instead of assumed to be zero |
| 3 | 2026-11-10 | Multi-step agent calls re-send a growing transcript (cost) | Automatic prompt caching (`cache_control`), with cache reads and writes counted separately in the cost | Cost per question is reported correctly in E6 (measure after the first API run) |
| 4 | 2026-11-10 | Smoke run: trajectory logs stored `"qid": ""` because the empty answer field overwrote the real id | Set `qid` and `run` on the answer before logging | Logs identify their question again |
| 5 | 2026-11-10 | `analyze_qa.py --root` still wrote its summaries into the real `results/`, mixing mock numbers with real results | Outputs follow the chosen root; the mock outputs were deleted | `results/` contains only real experiment outputs |
| 6 | 2026-11-10 | First Gap 2 run: requests had 93 "source" files at the test snapshot, 78 of them vendored (`requests/packages/urllib3`, `chardet`). Found by inspecting the file list, not the scores | Excluded vendored directories from the hotspot/risk population (only requests is affected) | requests test population 93 → 15 files. Mean P@10 difference hotspot − LOC changed from +5.0 pp to −2.5 pp. Both runs are kept (`results/gap2/pre_vendor_fix/`) |
| 7 | 2026-11-10 | Gap 2/3 test windows are long (5–9 years), so 60–72% of files are positive and P@10 saturates (xarray and pytest = 1.0 for every ranking) | Not changed (that would mean tuning on the test set); recorded as a threat to validity | Future work: fixed 6-month windows and function-level units, evaluated on new splits |
| 8 | 2026-11-10 | Risk explanations show collinearity (`fan_in` negative while `dependents_2hop` is positive in the same explanation) | Not changed yet; the `get_risk` text says it is a statistical estimate | Candidate fix: drop or merge correlated graph features after a tune-split check |
| 9 | 2026-11-11 | First real `ask` run without credentials crashed with an SDK traceback; in `run_qa.py` the same error would have been logged once per question (172 failed rows) | Missing/rejected credentials raise `MissingCredentials`, which stops the runner immediately with a one-line message | The CLI and runner exit cleanly with instructions; no failed rows are written |
| 10 | 2026-11-11 | First dev-split run with Gemini (`gemini-2.5-flash`): 20/20 questions failed with HTTP 400 on the first call; 0 tool calls, 0 judged answers. The provider's error text was hidden, failures were cached, and failed questions were marked done so a rerun would skip them | Show the provider's full error body; treat Gemini's "API key" 400 as a credentials error; never cache failed calls; failed questions are not written and are retried on resume; stop after 3 consecutive API failures | A broken model/key now stops the run after 3 questions with the real error message; 2 new tests (31 total). The failed run was discarded, not reported |

## Template for QA iterations (fill in from the dev split only)

| # | Date | Failure seen in dev trajectories (with log path) | Change | Effect on dev score / tokens (3 runs) |
|---|---|---|---|---|
