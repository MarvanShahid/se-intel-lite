"""Gap 1 / E1-E3 / E5 / E6 runner: run a system on SWE-QA questions, judge, save results and trajectories.

Outputs (resumable: finished (qid, run) pairs are skipped):
  results/qa/<system>/run<r>.jsonl     one line per question: scores, tokens, cost, latency, reliability fields
  logs/<system>/run<r>/<qid>.json      full trajectory (thoughts, tool calls, observation previews, answer)
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from ..agent.factory import SYSTEMS, SystemConfig, build_system, RepoContext
from ..llm.base import MissingCredentials, cost_usd, make_llm
from .dataset import ROOT, is_multi_hop, load_questions, reference_entities
from .judge import Judge

DEFAULT_LLM = os.environ.get("SEINTEL_LLM", "anthropic:claude-opus-5-5")
DEFAULT_JUDGE = os.environ.get("SEINTEL_JUDGE", "anthropic:claude-opus-5-5")


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["qid"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def evidence_recall(evidence: list[str], ref_entities: set[str], rg) -> float | None:
    """Share of files named by the reference answer that appear in the system's evidence (deterministic)."""
    ref_files = {rg.file_of(e) for e in ref_entities} - {None}
    if not ref_files:
        return None
    ev_files = {rg.file_of(e) for e in evidence if rg.has(e)} - {None}
    return len(ref_files & ev_files) / len(ref_files)


def run_system(system: str | SystemConfig, split: str = "test", runs: int = 3, llm_spec: str | None = None,
               judge_spec: str | None = None, repos: list[str] | None = None, limit: int | None = None,
               out_root: Path | None = None, judge: bool = True, verbose: bool = True) -> list[Path]:
    cfg = SYSTEMS[system] if isinstance(system, str) else system
    llm_spec = cfg.llm or llm_spec or DEFAULT_LLM
    judge_spec = judge_spec or DEFAULT_JUDGE
    out_root = out_root or ROOT
    questions = load_questions(split, repos, limit)
    judge_llm = Judge(make_llm(judge_spec, ROOT / ".cache" / "llm", run=0)) if judge else None
    outs = []
    for r in range(runs):
        out = out_root / "results" / "qa" / split / cfg.name / f"run{r}.jsonl"
        logd = out_root / "logs" / split / cfg.name / f"run{r}"
        out.parent.mkdir(parents=True, exist_ok=True)
        logd.mkdir(parents=True, exist_ok=True)
        done = _done(out)
        llm = make_llm(llm_spec, ROOT / ".cache" / "llm", run=r)
        model = llm.name
        for q in questions:
            if q["qid"] in done:
                continue
            try:
                sysobj = build_system(cfg, llm, q["repo"])
                ans = sysobj.run(q["question"], q["repo"])
                ans.qid, ans.run = q["qid"], r
                err = None
            except MissingCredentials:
                raise                # stop at once: every remaining question would fail the same way
            except Exception as e:   # one failing question must not kill the experiment
                ans, err = None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
            rg = RepoContext.get(q["repo"]).rg
            ref_ents = reference_entities(q["answer"], rg)
            row = {"qid": q["qid"], "repo": q["repo"], "system": cfg.name, "run": r, "llm": model,
                   "qtype": q["qtype"], "multi_hop": is_multi_hop(q["answer"], rg), "error": err}
            if ans is not None:
                row.update(answer=ans.answer, evidence=ans.evidence, tool_calls=ans.tool_calls, steps=ans.steps,
                           input_tokens=ans.input_tokens, output_tokens=ans.output_tokens,
                           cache_read=ans.cache_read, cache_write=ans.cache_write,
                           total_tokens=ans.input_tokens + ans.output_tokens + ans.cache_read + ans.cache_write,
                           cost_usd=cost_usd(model, ans.input_tokens, ans.output_tokens, ans.cache_read,
                                             ans.cache_write),
                           latency_s=round(ans.latency_s, 2), budget_exhausted=ans.budget_exhausted,
                           revisions=ans.revisions, cited=ans.cited, ungrounded=ans.ungrounded,
                           invalid_json=sum(1 for t in ans.trace if t.get("error") == "invalid_json"),
                           llm_errors=sum(1 for t in ans.trace if t.get("error") == "llm_error"),
                           evidence_recall=evidence_recall(ans.evidence, ref_ents, rg))
                if judge_llm:
                    row["scores"] = judge_llm.score(q["question"], q["answer"], ans.answer)
                (logd / f"{q['qid']}.json").write_text(json.dumps(
                    {"config": cfg.to_dict(), "qid": q["qid"], "question": q["question"], "reference": q["answer"],
                     **ans.to_dict(), "scores": row.get("scores")}, indent=2, ensure_ascii=False), encoding="utf-8")
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if verbose:
                sc = (row.get("scores") or {}).get("overall")
                print(f"[{cfg.name} run{r}] {q['qid']} score={sc} tools={row.get('tool_calls')} "
                      f"tok={row.get('total_tokens')} err={'yes' if err else 'no'}", flush=True)
        outs.append(out)
    return outs


def load_results(system: str, split: str = "test", root: Path | None = None) -> list[dict]:
    d = (root or ROOT) / "results" / "qa" / split / system
    rows = []
    for f in sorted(d.glob("run*.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows
