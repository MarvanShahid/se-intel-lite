"""The three LocAgent style graph tools [1]: SearchEntity, TraverseGraph, RetrieveEntity."""
from __future__ import annotations

from collections import deque

from ..agent.state import ToolResult
from ..graph.store import RepoGraph
from ..index.entity_index import EntityIndex
from .base import err, tool_spec

EDGE_LABEL = {("invokes", "out"): "invokes", ("invokes", "in"): "invoked_by",
              ("imports", "out"): "imports", ("imports", "in"): "imported_by",
              ("inherits", "out"): "inherits", ("inherits", "in"): "inherited_by",
              ("contains", "out"): "contains", ("contains", "in"): "contained_in"}
MAX_RETRIEVE_LINES = 80


def _brief(rg: RepoGraph, nid: str) -> str:
    d = rg.node(nid)
    if d["type"] == "directory":
        return f"[directory] {nid}"
    loc = f"L{d['start_line']}-{d['end_line']}"
    first = ""
    if d["type"] in ("function", "class"):
        first = rg.lines(d["path"])[d["start_line"] - 1].strip() if d["start_line"] - 1 < len(rg.lines(d["path"])) else ""
    return f"[{d['type']}] {nid} ({loc})" + (f"  {first[:100]}" if first else "")


class GraphTools:
    def __init__(self, rg: RepoGraph, index: EntityIndex | None = None, max_hops: int = 3):
        self.rg = rg
        self.index = index or EntityIndex(rg)
        self.max_hops = max_hops          # ablation A2: 1 vs 3

    # ------------------------------------------------------------ SearchEntity
    @tool_spec("search_entity",
               "Find files, classes or functions by name, id or keywords (layers: exact id, exact name, "
               "BM25 over ids, BM25 over code). Start here.",
               {"query": "str: identifier or keywords", "entity_type": "optional: file|class|function",
                "top_k": "optional int, default 8"})
    def search_entity(self, query: str, entity_type: str | None = None, top_k: int = 8) -> ToolResult:
        if not query or not str(query).strip():
            return err("query is empty")
        types = [entity_type] if entity_type in ("file", "class", "function") else None
        hits = self.index.search(str(query), types, int(top_k))
        if not hits:
            return ToolResult(True, f"No entities found for '{query}'. Try other keywords or identifiers.")
        lines = [f"Found {len(hits)} entities for '{query}':"]
        for nid, layer, _ in hits:
            lines.append(f"- {_brief(self.rg, nid)}  <{layer}>")
        return ToolResult(True, "\n".join(lines), [h[0] for h in hits],
                          {"layers": {h[0]: h[1] for h in hits}})

    # ----------------------------------------------------------- TraverseGraph
    @tool_spec("traverse_graph",
               "Breadth first walk from start entities over typed edges. Use it to find callers, callees, "
               "subclasses, importers and contained members. Hop limit 1-3.",
               {"start": "list[str] entity ids", "direction": "out|in|both (default out)",
                "edge_types": "list of invokes|imports|inherits|contains (default invokes,inherits,imports)",
                "hops": "int 1-3 (default 2)", "entity_types": "optional list of file|class|function",
                "max_nodes": "optional int (default 30)"})
    def traverse_graph(self, start, direction: str = "out", edge_types=None, hops: int = 2,
                       entity_types=None, max_nodes: int = 30) -> ToolResult:
        start = [start] if isinstance(start, str) else list(start or [])
        start = [s for s in start if self.rg.has(s)]
        if not start:
            return err("none of the start entities exist; use search_entity first")
        hops = max(1, min(int(hops), self.max_hops))
        etypes = set(edge_types or ["invokes", "inherits", "imports"])
        dirs = ["out", "in"] if direction == "both" else [direction if direction in ("out", "in") else "out"]
        keep = set(entity_types) if entity_types else None
        g = self.rg.g
        seen = set(start)
        order: list[tuple[int, str, str, str, bool]] = []  # depth, src, label, dst, heuristic
        q = deque((s, 0) for s in start)
        while q and len(seen) < max_nodes + len(start):
            cur, depth = q.popleft()
            if depth >= hops:
                continue
            for dr in dirs:
                edges = g.out_edges(cur, data=True) if dr == "out" else g.in_edges(cur, data=True)
                for u, v, e in edges:
                    nxt = v if dr == "out" else u
                    if e["type"] not in etypes or nxt in seen:
                        continue
                    if keep and g.nodes[nxt]["type"] not in keep and g.nodes[nxt]["type"] != "directory":
                        continue
                    seen.add(nxt)
                    order.append((depth + 1, cur, EDGE_LABEL[(e["type"], dr)], nxt, bool(e.get("heuristic"))))
                    q.append((nxt, depth + 1))
                    if len(seen) >= max_nodes + len(start):
                        break
        if not order:
            return ToolResult(True, "No neighbours found with these edge types / direction / hops.", start)
        lines = [f"Traversal from {', '.join(start)} (direction={direction}, hops={hops}):"]
        for depth, src, label, dst, heur in order:
            lines.append(f"{'  ' * (depth - 1)}{src} --{label}{'~' if heur else ''}--> {_brief(self.rg, dst)}")
        lines.append("(~ = heuristic edge: method name is unique in the repository but receiver type is unknown)")
        ents = start + [o[3] for o in order]
        return ToolResult(True, "\n".join(lines), ents, {"edges": [(o[1], o[2], o[3]) for o in order]})

    # ---------------------------------------------------------- RetrieveEntity
    @tool_spec("retrieve_entity",
               "Read the source code of entities (max 5 per call). For a file you get an outline "
               "of its classes and functions; for a class or function you get its numbered code.",
               {"entities": "list[str] entity ids"})
    def retrieve_entity(self, entities) -> ToolResult:
        ents = [entities] if isinstance(entities, str) else list(entities or [])
        if not ents:
            return err("entities is empty")
        out, got = [], []
        for nid in ents[:5]:
            if not self.rg.has(nid):
                out.append(f"## {nid}\nNot found. Use search_entity to find the right id.")
                continue
            d = self.rg.node(nid)
            got.append(nid)
            if d["type"] == "directory":
                kids = [c for _, c, e in self.rg.g.out_edges(nid, data=True) if e["type"] == "contains"]
                out.append(f"## {nid} (directory)\n" + "\n".join(f"- {k}" for k in sorted(kids)[:50]))
            elif d["type"] == "file":
                kids = [c for _, c, e in self.rg.g.out_edges(nid, data=True) if e["type"] == "contains"]
                out.append(f"## {nid} (file, {d['end_line']} lines) outline:\n" +
                           "\n".join(f"- {_brief(self.rg, k)}" for k in sorted(kids, key=lambda k: self.rg.node(k)["start_line"])[:60]))
            else:
                ls = self.rg.lines(d["path"])[d["start_line"] - 1: d["end_line"]]
                trunc = len(ls) > MAX_RETRIEVE_LINES
                body = "\n".join(f"{d['start_line'] + i:>5}  {l}" for i, l in enumerate(ls[:MAX_RETRIEVE_LINES]))
                out.append(f"## {nid} ({d['type']}, L{d['start_line']}-{d['end_line']})\n{body}"
                           + ("\n... (truncated; retrieve inner functions for the rest)" if trunc else ""))
        return ToolResult(True, "\n\n".join(out), got, {"retrieved": got})
