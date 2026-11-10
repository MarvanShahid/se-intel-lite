"""Static quality metrics (radon) and file-level graph features."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

from ..graph.store import RepoGraph


def static_metrics(src: str) -> dict:
    """radon metrics for one file's source; on failure returns size-only metrics plus 'error'."""
    m: dict = {"loc": src.count("\n") + 1 if src else 0}
    try:
        raw = analyze(src)
        blocks = cc_visit(src)
        ccs = [b.complexity for b in blocks]
        m.update(loc=raw.loc, sloc=raw.sloc, comments=raw.comments, mi=round(mi_visit(src, multi=True), 1),
                 n_blocks=len(ccs), cc_max=max(ccs, default=0), cc_sum=sum(ccs),
                 cc_mean=round(sum(ccs) / len(ccs), 2) if ccs else 0.0,
                 blocks={(f"{b.classname}.{b.name}" if getattr(b, "classname", None) else b.name): b.complexity
                         for b in blocks})
    except Exception as e:  # radon can fail on unusual syntax
        m.update(sloc=m["loc"], comments=0, mi=100.0, n_blocks=0, cc_max=0, cc_sum=0, cc_mean=0.0, blocks={},
                 error=str(e))
    return m


def file_dependency_graph(rg: RepoGraph, include_heuristic: bool = False) -> nx.DiGraph:
    """Collapse the code graph to files: edge f -> g when f imports g or an entity in f invokes/inherits one in g."""
    fg = nx.DiGraph()
    for nid, d in rg.g.nodes(data=True):
        if d["type"] == "file":
            fg.add_node(nid)
    for u, v, e in rg.g.edges(data=True):
        if e["type"] not in ("imports", "invokes", "inherits"):
            continue
        if e.get("heuristic") and not include_heuristic:
            continue
        fu, fv = rg.file_of(u), rg.file_of(v)
        if fu and fv and fu != fv:
            if fg.has_edge(fu, fv):
                fg[fu][fv]["w"] += 1
            else:
                fg.add_edge(fu, fv, w=1)
    return fg


def graph_features(rg: RepoGraph) -> dict[str, dict]:
    fg = file_dependency_graph(rg)
    pr = nx.pagerank(fg, weight="w") if fg.number_of_edges() else {n: 1 / max(1, len(fg)) for n in fg}
    rev = fg.reverse(copy=False)
    out = {}
    calls_in = defaultdict(int)
    for u, v, e in rg.g.edges(data=True):
        if e["type"] == "invokes" and not e.get("heuristic"):
            fu, fv = rg.file_of(u), rg.file_of(v)
            if fu and fv and fu != fv:
                calls_in[fv] += 1
    for f in fg.nodes:
        two_hop = set()
        for n in rev.successors(f):
            two_hop.add(n)
            two_hop.update(rev.successors(n))
        two_hop.discard(f)
        out[f] = {"fan_in": fg.in_degree(f), "fan_out": fg.out_degree(f), "pagerank": pr.get(f, 0.0),
                  "dependents_2hop": len(two_hop), "calls_in": calls_in[f]}
    return out
