"""Gap 2 / Gap 3 tools: RunQuality, GetHotspots and GetRisk.

All three are read-only. GetHotspots and GetRisk read the per-file table computed at the pinned commit
(risk/features.build_current_table) and the artefacts produced by experiments/gap2_hotspots.py
(tuned weights) and experiments/gap3_risk.py (calibrated model).
"""
from __future__ import annotations

import json
import shutil
import subprocess

from ..agent.state import ToolResult
from ..eval.dataset import ROOT
from ..graph.store import RepoGraph
from ..quality import hotspot
from ..quality.metrics import static_metrics
from .base import err, tool_spec

RISK_MODEL = ROOT / "results" / "gap3" / "risk_model_full.json"
HOTSPOT_WEIGHTS = ROOT / "results" / "gap2" / "weights.json"


def _pylint(path) -> dict | None:
    """Optional pylint message counts (only if pylint is installed); never raises."""
    exe = shutil.which("pylint")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--output-format=json", "--disable=C0114,C0115,C0116", "--score=n", str(path)],
                           capture_output=True, text=True, timeout=60)
        msgs = json.loads(r.stdout or "[]")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None
    out: dict[str, int] = {}
    for m in msgs:
        out[m.get("type", "other")] = out.get(m.get("type", "other"), 0) + 1
    return out


class QualityTools:
    def __init__(self, rg: RepoGraph, repo: str | None = None, rows: list[dict] | None = None,
                 use_pylint: bool = False):
        self.rg, self.repo, self.use_pylint = rg, repo, use_pylint
        self._rows = rows
        self._file_cache: dict[str, dict] = {}
        self._risk = None

    # ------------------------------------------------------------------ data
    @property
    def rows(self) -> list[dict]:
        if self._rows is None:
            from ..risk.features import build_current_table
            self._rows = build_current_table(self.repo) if self.repo else []
        return self._rows

    def _row(self, path: str) -> dict | None:
        return next((r for r in self.rows if r["path"] == path), None)

    def _weights(self):
        if HOTSPOT_WEIGHTS.exists():
            w = json.loads(HOTSPOT_WEIGHTS.read_text())
            return (w["w_in"], w["w_out"])
        return hotspot.DEFAULT_WEIGHTS

    def file_metrics(self, path: str) -> dict:
        if path not in self._file_cache:
            self._file_cache[path] = {"path": path, **static_metrics("\n".join(self.rg.lines(path)))}
        return self._file_cache[path]

    # ------------------------------------------------------------ RunQuality
    @tool_spec("run_quality",
               "Static quality metrics for a file, class or function: lines, cyclomatic complexity, maintainability "
               "index, and for files the graph fan-in/fan-out and hotspot rank.",
               {"entity": "str entity id"})
    def run_quality(self, entity: str) -> ToolResult:
        if not self.rg.has(entity):
            return err(f"unknown entity {entity}; use search_entity first")
        d = self.rg.node(entity)
        if d["type"] == "directory":
            return err("run_quality needs a file, class or function")
        fm = self.file_metrics(d["path"])
        if d["type"] == "file":
            top = sorted(fm["blocks"].items(), key=lambda kv: -kv[1])[:5]
            txt = (f"{entity}: loc={fm['loc']} sloc={fm['sloc']} maintainability_index={fm['mi']} "
                   f"cc_sum={fm['cc_sum']} max_cc={fm['cc_max']} mean_cc={fm['cc_mean']}; most complex: "
                   + (", ".join(f"{k}={v}" for k, v in top) or "none"))
            row = self._row(entity)
            if row:
                ranked = hotspot.rank(self.rows, "hotspot", self._weights())
                txt += (f"; fan_in={row['fan_in']} fan_out={row['fan_out']} "
                        f"hotspot_rank={ranked.index(entity) + 1}/{len(ranked)}")
            if self.use_pylint:
                pl = _pylint(self.rg.root / entity)
                if pl is not None:
                    txt += "; pylint " + (", ".join(f"{k}={v}" for k, v in sorted(pl.items())) or "clean")
        else:
            cc = fm["blocks"].get(d["qualname"])
            n = d["end_line"] - d["start_line"] + 1
            txt = f"{entity}: lines={n} cyclomatic_complexity={cc if cc is not None else 'n/a'}"
        if "error" in fm:
            txt += f" (radon partial: {fm['error'][:80]})"
        return ToolResult(True, txt, [entity], {"metrics": {k: v for k, v in fm.items() if k != "blocks"}})

    # ----------------------------------------------------------- GetHotspots
    @tool_spec("get_hotspots",
               "Rank source files (tests excluded) by the graph-aware hotspot score (complexity + how many files "
               "depend on it + how many it depends on). Optional directory prefix filter.",
               {"top_k": "optional int, default 10", "prefix": "optional str path prefix, e.g. 'src/flask'"})
    def get_hotspots(self, top_k: int = 10, prefix: str | None = None) -> ToolResult:
        rows = [r for r in self.rows if not prefix or r["path"].startswith(prefix)]
        if not rows:
            return err("no source files found (check the prefix)")
        ranked = hotspot.rank(rows, "hotspot", self._weights())
        by = {r["path"]: r for r in rows}
        k = max(1, min(int(top_k), 25))
        lines = [f"Top {k} hotspots of {len(rows)} source files:"]
        lines += [f"{i + 1}. {hotspot.explain(by[p], i + 1, len(rows))}" for i, p in enumerate(ranked[:k])]
        return ToolResult(True, "\n".join(lines), ranked[:k], {"ranking": ranked[:k]})

    # --------------------------------------------------------------- GetRisk
    @tool_spec("get_risk",
               "Calibrated probability that a source file receives a bug-fix commit in the next period, with the "
               "features that drive it. Use for questions about risk, fragility or safety of changes.",
               {"entity": "str file id (or any entity id inside the file)"})
    def get_risk(self, entity: str) -> ToolResult:
        path = self.rg.file_of(entity) if self.rg.has(entity) else None
        if not path:
            return err(f"unknown entity {entity}; use search_entity first")
        if not RISK_MODEL.exists():
            return err("risk model not trained yet; run experiments/gap3_risk.py")
        row = self._row(path)
        if row is None:
            return ToolResult(True, f"{path} is a test/doc/build file; the risk model covers source files only.",
                              [path], {"p_risky": None})
        if self._risk is None:
            from ..risk.model import RiskModel
            self._risk = RiskModel.from_json(json.loads(RISK_MODEL.read_text()))
        p = float(self._risk.predict([row])[0])
        all_p = self._risk.predict(self.rows)
        pct = float((all_p < p).mean() * 100)
        contrib = self._risk.contributions(row)
        why = ", ".join(f"{f}={v:g} ({'+' if c >= 0 else '-'}{abs(c):.2f})" for f, v, c in contrib)
        txt = (f"{path}: p(bug-fix in next period)={p:.2f} (calibrated={self._risk.platt is not None}); "
               f"higher than {pct:.0f}% of source files. Main drivers (logit contribution): {why}. "
               "This is a statistical estimate from history and structure, not a statement that a specific bug exists.")
        return ToolResult(True, txt, [path], {"p_risky": p, "calibrated": self._risk.platt is not None,
                                              "percentile": pct, "features": {f: v for f, v, _ in contrib}})
