"""Deterministic scripted LLM used by the tests and the `--llm mock` smoke run (no API key needed).

It follows a fixed policy: search the question keywords, traverse from the first hit, retrieve it, finish.
Its numbers are NOT research results; they only prove the pipeline runs end to end.
"""
from __future__ import annotations

import json
import re

from .base import LLM, LLMResponse

STOP = set("the a an of to in is are does do how what why where which when who and or for with on by this that "
           "it be as from its into can used use".split())


def _keywords(q: str, n: int = 4) -> str:
    words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", q) if w.lower() not in STOP]
    return " ".join(words[:n]) or q[:40]


class MockLLM(LLM):
    name = "mock"

    def __init__(self, script: list[str] | None = None, **_):
        self.script = list(script) if script else None

    def _complete(self, system, messages, max_tokens) -> LLMResponse:
        if self.script is not None:
            text = self.script.pop(0) if self.script else '{"action":"finish","args":{"answer":"done"}}'
            return LLMResponse(text, 10, 10)
        user0 = messages[0]["content"]
        q = user0.split("QUESTION:", 1)[-1].strip()
        if "AVAILABLE TOOLS" not in system:          # B0 / B1 / judge style single call
            if "SCORE" in system.upper() and "JSON" in system.upper():
                return LLMResponse(json.dumps({"correctness": 5, "completeness": 5, "relevance": 5,
                                               "clarity": 5, "reasoning": 5}), 10, 10)
            return LLMResponse(f"Mock answer about: {_keywords(q)}", 10, 10)
        obs = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("OBSERVATION")]
        ids = re.findall(r"\] (\S+\.py(?::[\w.]+)?) \(L", obs[-1]) if obs else []
        n = len(obs)
        if n == 0:
            act = {"thought": "search first", "action": "search_entity", "args": {"query": _keywords(q)}}
        elif n == 1 and ids:
            act = {"thought": "follow calls", "action": "traverse_graph",
                   "args": {"start": [ids[0]], "direction": "out", "hops": 2}}
        elif n == 2 and ids:
            act = {"thought": "read code", "action": "retrieve_entity", "args": {"entities": ids[:2]}}
        else:
            seen = re.findall(r"## (\S+) \(", "\n".join(obs))
            act = {"thought": "enough", "action": "finish",
                   "args": {"answer": f"Mock answer about {_keywords(q)}; see {', '.join(seen[:2]) or 'n/a'}.",
                            "evidence": seen[:2]}}
        return LLMResponse(json.dumps(act), 50, 20)
