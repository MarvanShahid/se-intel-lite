"""System configurations (baselines, proposed system, ablations) and a factory that builds them.

Every gap-specific component is a flag here, so each ablation is one config with one flag changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..eval.dataset import ROOT, repo_path
from ..graph.store import RepoGraph
from ..index.chunk_index import ChunkIndex
from ..index.entity_index import EntityIndex
from ..llm.base import LLM

GRAPH_TOOLS = ["search_entity", "traverse_graph", "retrieve_entity"]
QUALITY_TOOLS = ["run_quality", "get_hotspots"]      # Gap 2
RISK_TOOLS = ["get_risk"]                            # Gap 3


@dataclass(frozen=True)
class SystemConfig:
    name: str
    kind: str = "agent"                  # plain | rag | agent
    tools: tuple = tuple(GRAPH_TOOLS)
    max_hops: int = 3
    qa_prompt: bool = False              # Gap 1
    verify_citations: bool = False       # Gap 1 (robustness)
    max_steps: int = 12
    rag_top_k: int = 8
    llm: str | None = None               # override the run's LLM (ablation A6)

    def to_dict(self) -> dict:
        return asdict(self)


P = SystemConfig("P", tools=tuple(GRAPH_TOOLS + QUALITY_TOOLS + RISK_TOOLS), qa_prompt=True, verify_citations=True)

SYSTEMS: dict[str, SystemConfig] = {
    "B0": SystemConfig("B0", kind="plain", tools=()),
    "B1": SystemConfig("B1", kind="rag", tools=()),
    "B2": SystemConfig("B2"),                                   # LocAgent-style base paper approach [1]
    "P": P,
    # E5 ablations of P (one component removed at a time)
    "A1_no_traverse": replace(P, name="A1_no_traverse", tools=tuple(t for t in P.tools if t != "traverse_graph")),
    "A2_hop1": replace(P, name="A2_hop1", max_hops=1),
    "A3_no_quality": replace(P, name="A3_no_quality", tools=tuple(t for t in P.tools if t not in QUALITY_TOOLS)),
    "A3b_no_risk": replace(P, name="A3b_no_risk", tools=tuple(t for t in P.tools if t not in RISK_TOOLS)),
    "A7_no_qa_prompt": replace(P, name="A7_no_qa_prompt", qa_prompt=False),
    "A8_no_verifier": replace(P, name="A8_no_verifier", verify_citations=False),
    "A6_small_llm": replace(P, name="A6_small_llm", llm="anthropic:claude-haiku-5-5"),
}


class RepoContext:
    """Per-repository graph, indexes and tool objects (built once, shared by all systems)."""
    _cache: dict[str, "RepoContext"] = {}

    def __init__(self, repo: str):
        self.repo = repo
        self.rg = RepoGraph.from_repo(repo_path(repo), cache_dir=ROOT / ".cache" / "graphs" / "pinned")
        self.entity_index = EntityIndex(self.rg)
        self._chunks: ChunkIndex | None = None

    @classmethod
    def get(cls, repo: str) -> "RepoContext":
        if repo not in cls._cache:
            cls._cache[repo] = cls(repo)
        return cls._cache[repo]

    @property
    def chunks(self) -> ChunkIndex:
        if self._chunks is None:
            self._chunks = ChunkIndex(self.rg)
        return self._chunks


def build_system(cfg: SystemConfig, llm: LLM, repo: str):
    from ..baselines.simple import FlatRAG, PlainLLM
    from ..tools.graph_tools import GraphTools
    from ..tools.quality_tools import QualityTools
    from .react import ReActAgent

    if cfg.kind == "plain":
        return PlainLLM(llm)
    ctx = RepoContext.get(repo)
    if cfg.kind == "rag":
        return FlatRAG(llm, ctx.chunks, cfg.rag_top_k)
    gt = GraphTools(ctx.rg, ctx.entity_index, max_hops=cfg.max_hops)
    qt = QualityTools(ctx.rg, repo)
    available = {"search_entity": gt.search_entity, "traverse_graph": gt.traverse_graph,
                 "retrieve_entity": gt.retrieve_entity, "run_quality": qt.run_quality,
                 "get_hotspots": qt.get_hotspots, "get_risk": qt.get_risk}
    tools = {t: available[t] for t in cfg.tools}
    return ReActAgent(llm, tools, cfg.name, max_steps=cfg.max_steps, qa_prompt=cfg.qa_prompt,
                      verify_citations=cfg.verify_citations)
