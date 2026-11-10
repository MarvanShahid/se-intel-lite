"""Structured contracts shared by the agent, tools, baselines and the evaluation harness."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    text: str                                        # observation shown to the LLM
    entities: list[str] = field(default_factory=list)  # entity ids this call surfaced
    data: dict[str, Any] = field(default_factory=dict)  # machine readable payload


@dataclass
class ToolCall:
    step: int
    tool: str
    args: dict
    ok: bool
    obs_chars: int
    entities: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    question: str
    repo: str
    step: int = 0
    max_steps: int = 12
    visited: set = field(default_factory=set)          # entity ids seen in any observation
    retrieved: set = field(default_factory=set)        # entity ids whose code was actually read
    tool_log: list[ToolCall] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    budget_exhausted: bool = False


@dataclass
class AgentAnswer:
    system: str                       # B0 | B1 | B2 | P
    repo: str
    question: str
    answer: str
    evidence: list[str]               # entity ids cited / used as context
    tool_calls: int = 0
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    budget_exhausted: bool = False
    llm: str = ""
    trace: list[dict] = field(default_factory=list)
    revisions: int = 0                # citation-verifier revisions requested (Gap 1)
    cited: int = 0                    # ids the model cited in finish
    ungrounded: int = 0               # cited ids that never appeared in an observation
    cache_read: int = 0
    cache_write: int = 0
    qid: str = ""
    run: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
