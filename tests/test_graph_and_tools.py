from seintel.graph.store import RepoGraph
from seintel.index.chunk_index import ChunkIndex
from seintel.index.entity_index import EntityIndex, tokenize
from seintel.quality.metrics import file_dependency_graph, graph_features
from seintel.tools.graph_tools import GraphTools


def edges(rg, t):
    return {(u, v) for u, v, d in rg.g.edges(data=True) if d["type"] == t}


def test_graph_schema(tiny_repo):
    rg = RepoGraph.from_repo(tiny_repo)
    assert rg.node("pkg/core.py:Engine")["type"] == "class"
    assert ("pkg/core.py:Engine", "pkg/base.py:Base") in edges(rg, "inherits")
    assert ("pkg/core.py", "pkg/util.py") in edges(rg, "imports")
    inv = edges(rg, "invokes")
    assert ("pkg/core.py:Engine.step", "pkg/util.py:helper") in inv
    # self.run() resolves through inheritance to Base.run
    assert ("pkg/core.py:Engine.dispatch_request", "pkg/base.py:Base.run") in inv


def test_tokenize_splits_identifiers():
    assert tokenize("dispatch_request getRepoGraph") == ["dispatch", "request", "get", "repo", "graph"]


def test_search_layers(tiny_repo):
    rg = RepoGraph.from_repo(tiny_repo)
    idx = EntityIndex(rg)
    hits = idx.search("pkg/util.py:helper")
    assert hits[0][:2] == ("pkg/util.py:helper", "exact_id")
    assert idx.search("dispatch_request")[0][0] == "pkg/core.py:Engine.dispatch_request"


def test_traverse_hop_limit(tiny_repo):
    rg = RepoGraph.from_repo(tiny_repo)
    t1 = GraphTools(rg, max_hops=1).traverse_graph(["pkg/core.py:Engine.dispatch_request"], hops=3)
    t3 = GraphTools(rg, max_hops=3).traverse_graph(["pkg/core.py:Engine.dispatch_request"], hops=3)
    assert set(t1.entities) < set(t3.entities)        # A2 ablation really restricts the walk
    assert "hops=1" in t1.text


def test_retrieve_and_unknown(tiny_repo):
    gt = GraphTools(RepoGraph.from_repo(tiny_repo))
    r = gt.retrieve_entity(["pkg/util.py:helper", "nope.py:x"])
    assert r.ok and r.data["retrieved"] == ["pkg/util.py:helper"] and "Not found" in r.text
    assert not gt.traverse_graph(["missing"]).ok


def test_chunk_index(tiny_repo):
    ci = ChunkIndex(RepoGraph.from_repo(tiny_repo))
    assert ci.search("helper doubled")[0][0] == "pkg/util.py:helper"


def test_file_graph_features(tiny_repo):
    rg = RepoGraph.from_repo(tiny_repo)
    fg = file_dependency_graph(rg)
    assert fg.has_edge("pkg/core.py", "pkg/base.py")
    gf = graph_features(rg)
    assert gf["pkg/base.py"]["fan_in"] >= 1 and gf["pkg/util.py"]["fan_out"] == 0
