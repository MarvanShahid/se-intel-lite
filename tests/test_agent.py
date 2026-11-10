import json

from seintel.agent.react import ReActAgent, _parse_action
from seintel.graph.store import RepoGraph
from seintel.llm.mock import MockLLM
from seintel.tools.graph_tools import GraphTools


def _agent(repo, script, **kw):
    gt = GraphTools(RepoGraph.from_repo(repo))
    tools = {"search_entity": gt.search_entity, "traverse_graph": gt.traverse_graph,
             "retrieve_entity": gt.retrieve_entity}
    return ReActAgent(MockLLM(script=[json.dumps(s) if isinstance(s, dict) else s for s in script]), tools, "T", **kw)


def test_parse_action_tolerates_fences_and_prose():
    assert _parse_action('Sure!\n```json\n{"thought": "x", "action": "finish", "args": {}}\n```')["action"] == "finish"
    assert _parse_action("no json here") is None


def test_full_loop_and_evidence(tiny_repo):
    a = _agent(tiny_repo, [
        {"action": "search_entity", "args": {"query": "helper"}},
        {"action": "retrieve_entity", "args": {"entities": ["pkg/util.py:helper"]}},
        {"action": "finish", "args": {"answer": "helper doubles n.", "evidence": ["pkg/util.py:helper", "fake.py:x"]}}])
    ans = a.run("What does helper do?", "tiny")
    assert ans.answer == "helper doubles n." and ans.tool_calls == 2
    assert ans.evidence == ["pkg/util.py:helper"] and ans.ungrounded == 1     # fake id is dropped and counted


def test_invalid_json_and_bad_tool_recover(tiny_repo):
    a = _agent(tiny_repo, ["garbage", {"action": "nope", "args": {}},
                           {"action": "finish", "args": {"answer": "x"}}])
    ans = a.run("q", "tiny")
    assert ans.answer == "x" and any(t.get("error") == "invalid_json" for t in ans.trace)


def test_step_budget(tiny_repo):
    loop = [{"action": "search_entity", "args": {"query": "helper"}}] * 10
    ans = _agent(tiny_repo, loop, max_steps=3).run("q", "tiny")
    assert ans.budget_exhausted and ans.tool_calls == 3


def test_citation_verifier_requests_revision(tiny_repo):
    script = [{"action": "search_entity", "args": {"query": "helper"}},
              {"action": "finish", "args": {"answer": "short", "evidence": ["made/up.py:f"]}},
              {"action": "retrieve_entity", "args": {"entities": ["pkg/util.py:helper"]}},
              {"action": "finish", "args": {"answer": "helper in pkg/util.py returns its argument doubled (n * 2).",
                                            "evidence": ["pkg/util.py:helper"]}}]
    on = _agent(tiny_repo, script, verify_citations=True).run("q", "tiny")
    assert on.revisions == 1 and on.ungrounded == 0 and "doubled" in on.answer
    off = _agent(tiny_repo, script, verify_citations=False).run("q", "tiny")
    assert off.revisions == 0 and off.answer == "short"


def test_mock_policy_end_to_end(tiny_repo):
    gt = GraphTools(RepoGraph.from_repo(tiny_repo))
    tools = {"search_entity": gt.search_entity, "traverse_graph": gt.traverse_graph,
             "retrieve_entity": gt.retrieve_entity}
    ans = ReActAgent(MockLLM(), tools, "T").run("How does dispatch_request work?", "tiny")
    assert ans.tool_calls == 3 and ans.evidence
