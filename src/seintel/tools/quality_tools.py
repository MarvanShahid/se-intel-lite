"""RunQuality (initial implementation, radon) and GetRisk (typed stub, filled in Phase 4)."""
from __future__ import annotations

from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

from ..agent.state import ToolResult
from ..graph.store import RepoGraph
from .base import err, tool_spec


class QualityTools:
    def __init__(self, rg: RepoGraph):
        self.rg = rg
        self._file_cache: dict[str, dict] = {}

    def file_metrics(self, path: str) -> dict:
        if path in self._file_cache:
            return self._file_cache[path]
        src = "\n".join(self.rg.lines(path))
        m: dict = {"path": path}
        try:
            raw = analyze(src)
            blocks = cc_visit(src)
            ccs = [b.complexity for b in blocks]
            m.update(loc=raw.loc, sloc=raw.sloc, comments=raw.comments,
                     mi=round(mi_visit(src, multi=True), 1),
                     n_blocks=len(ccs), cc_max=max(ccs, default=0),
                     cc_mean=round(sum(ccs) / len(ccs), 2) if ccs else 0.0,
                     blocks={b.name if not getattr(b, "classname", None) else f"{b.classname}.{b.name}": b.complexity
                             for b in blocks})
        except Exception as e:  # radon can fail on odd syntax
            m["error"] = str(e)
        self._file_cache[path] = m
        return m

    @tool_spec("run_quality",
               "Static quality metrics (radon): lines of code, cyclomatic complexity, maintainability index "
               "for a file, class or function.",
               {"entity": "str entity id"})
    def run_quality(self, entity: str) -> ToolResult:
        if not self.rg.has(entity):
            return err(f"unknown entity {entity}")
        d = self.rg.node(entity)
        if d["type"] == "directory":
            return err("run_quality needs a file, class or function")
        fm = self.file_metrics(d["path"])
        if "error" in fm:
            return err(f"radon failed: {fm['error']}")
        if d["type"] == "file":
            top = sorted(fm["blocks"].items(), key=lambda kv: -kv[1])[:5]
            txt = (f"{entity}: loc={fm['loc']} sloc={fm['sloc']} maintainability_index={fm['mi']} "
                   f"max_cc={fm['cc_max']} mean_cc={fm['cc_mean']}; most complex: "
                   + ", ".join(f"{k}={v}" for k, v in top))
        else:
            cc = fm["blocks"].get(d["qualname"])
            n = d["end_line"] - d["start_line"] + 1
            txt = f"{entity}: lines={n} cyclomatic_complexity={cc if cc is not None else 'n/a'}"
        return ToolResult(True, txt, [entity], {"metrics": fm})

    @tool_spec("get_risk",
               "[NOT AVAILABLE YET - Phase 4] Calibrated probability that a file receives a bug-fix commit "
               "in the next period.",
               {"entity": "str file id"})
    def get_risk(self, entity: str) -> ToolResult:
        # Contract is fixed now so the agent prompt and evaluation code do not change later:
        # data = {"p_risky": float in [0,1], "calibrated": bool, "features": {...}, "window": str}
        return ToolResult(False, "get_risk is not available: the risk model is trained in Phase 4.", [entity],
                          {"p_risky": None, "calibrated": False})
