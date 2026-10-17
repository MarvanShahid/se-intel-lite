"""Graph + source access, with on disk caching."""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from .parser import build_graph


class RepoGraph:
    """Wraps the NetworkX graph and gives read-only access to source text."""

    def __init__(self, g: nx.MultiDiGraph, root: str | Path):
        self.g = g
        self.root = Path(root)
        self._file_cache: dict[str, list[str]] = {}

    @classmethod
    def from_repo(cls, repo_root: str | Path, cache_dir: str | Path | None = None) -> "RepoGraph":
        repo_root = Path(repo_root).resolve()
        if cache_dir:
            cache = Path(cache_dir) / f"{repo_root.name}.graph.pkl"
            if cache.exists():
                with open(cache, "rb") as f:
                    return cls(pickle.load(f), repo_root)
        g = build_graph(repo_root)
        if g.number_of_nodes() <= 1:
            raise ValueError(f"no Python files found under {repo_root}")
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(g, f)
        return cls(g, repo_root)

    # ----------------------------------------------------------------- access
    def has(self, nid: str) -> bool:
        return nid in self.g

    def node(self, nid: str) -> dict:
        return self.g.nodes[nid]

    def lines(self, path: str) -> list[str]:
        if path not in self._file_cache:
            try:
                txt = (self.root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = ""
            self._file_cache[path] = txt.splitlines()
        return self._file_cache[path]

    def code(self, nid: str) -> str:
        d = self.g.nodes[nid]
        if d["type"] == "directory":
            return ""
        ls = self.lines(d["path"])
        return "\n".join(ls[d["start_line"] - 1: d["end_line"]])

    def entities(self, types=("file", "class", "function")):
        for nid, d in self.g.nodes(data=True):
            if d["type"] in types:
                yield nid, d

    def file_of(self, nid: str) -> str | None:
        d = self.g.nodes.get(nid)
        return d.get("path") if d and d["type"] != "directory" else None

    def stats(self) -> dict:
        from collections import Counter
        nt = Counter(d["type"] for _, d in self.g.nodes(data=True))
        et = Counter(d["type"] for _, _, d in self.g.edges(data=True))
        return {"nodes": dict(nt), "edges": dict(et), "builder": self.g.graph.get("stats", {})}
