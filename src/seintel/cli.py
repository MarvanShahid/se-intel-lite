"""Command line interface.

  python -m seintel ask --repo flask "How does Flask dispatch a request to a view function?"
  python -m seintel ask --repo /path/to/any/python/repo --system B2 "..."      (any local repository)
  python -m seintel hotspots --repo flask --top 10
  python -m seintel risk --repo flask src/flask/app.py
  python -m seintel graph --repo requests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_name(arg: str) -> str:
    from .eval.dataset import load_config
    names = [r["name"] for r in load_config()["repos"]]
    if arg in names:
        return arg
    raise SystemExit(f"--repo must be one of the configured repositories {names} for this command")


def cmd_ask(a):
    from .agent.factory import SYSTEMS, RepoContext, build_system
    from .eval.dataset import ROOT, load_config
    from .llm.base import cost_usd, make_llm
    names = [r["name"] for r in load_config()["repos"]]
    if a.repo not in names:   # ad-hoc local repository: register a context under its folder name
        from .graph.store import RepoGraph
        from .index.entity_index import EntityIndex
        p = Path(a.repo).resolve()
        ctx = RepoContext.__new__(RepoContext)
        ctx.repo, ctx.rg = p.name, RepoGraph.from_repo(p, ROOT / ".cache" / "graphs" / "adhoc")
        ctx.entity_index, ctx._chunks = EntityIndex(ctx.rg), None
        RepoContext._cache[p.name] = ctx
        repo = p.name
        if a.system == "P":
            print("note: hotspot/risk tools need the history tables, which exist only for configured repos; "
                  "they will report an error for ad-hoc repositories.", file=sys.stderr)
    else:
        repo = a.repo
    cfg = SYSTEMS[a.system]
    llm = make_llm(cfg.llm or a.llm)
    ans = build_system(cfg, llm, repo).run(a.question, repo)
    if a.json:
        print(json.dumps(ans.to_dict(), indent=2))
        return
    if a.trace:
        for t in ans.trace:
            print("  ", json.dumps(t)[:300])
    print("\n" + ans.answer + "\n")
    print("Evidence:", *ans.evidence[:15], sep="\n  - ")
    print(f"\n[{ans.system}] {ans.tool_calls} tool calls, {ans.input_tokens + ans.output_tokens} tokens, "
          f"${cost_usd(llm.name, ans.input_tokens, ans.output_tokens, ans.cache_read, ans.cache_write):.4f}, "
          f"{ans.latency_s:.1f}s{' (budget exhausted)' if ans.budget_exhausted else ''}")


def cmd_hotspots(a):
    from .agent.factory import RepoContext
    from .tools.quality_tools import QualityTools
    r = _repo_name(a.repo)
    print(QualityTools(RepoContext.get(r).rg, r).get_hotspots(a.top, a.prefix).text)


def cmd_risk(a):
    from .agent.factory import RepoContext
    from .tools.quality_tools import QualityTools
    r = _repo_name(a.repo)
    print(QualityTools(RepoContext.get(r).rg, r).get_risk(a.file).text)


def cmd_graph(a):
    from .agent.factory import RepoContext
    print(json.dumps(RepoContext.get(_repo_name(a.repo)).rg.stats(), indent=2))


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(prog="seintel", description="SE-INTEL Lite repository agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--repo", required=True, help="configured repo name or a path to a local Python repository")
    p.add_argument("--system", default="P", help="B0 | B1 | B2 | P | an ablation name")
    p.add_argument("--llm", default=os.environ.get("SEINTEL_LLM", "anthropic:claude-opus-5-5"))
    p.add_argument("--trace", action="store_true", help="print the agent trajectory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("hotspots")
    p.add_argument("--repo", required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--prefix", default=None)
    p.set_defaults(fn=cmd_hotspots)
    p = sub.add_parser("risk")
    p.add_argument("file")
    p.add_argument("--repo", required=True)
    p.set_defaults(fn=cmd_risk)
    p = sub.add_parser("graph")
    p.add_argument("--repo", required=True)
    p.set_defaults(fn=cmd_graph)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
