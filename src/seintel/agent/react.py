"""ReAct style agent with a step budget, typed tools and an evidence trail.

The same loop is used for baseline B2 (3 graph tools) and the proposed system P (+ RunQuality, GetRisk).
"""
from __future__ import annotations

import json
import re
import time

from ..llm.base import LLM
from ..tools.base import TOOL_SPECS
from .state import AgentAnswer, AgentState, ToolCall, ToolResult

SYSTEM = """You are a code intelligence agent that answers questions about ONE Python repository.
You cannot see the code directly: you explore it with tools, then answer.

AVAILABLE TOOLS (call one per turn):
{tools}
- finish: {{"answer": "str", "evidence": ["entity ids you actually read or saw"]}}

PROTOCOL
- Reply with exactly ONE JSON object and nothing else:
  {{"thought": "short reasoning", "action": "<tool name or finish>", "args": {{...}}}}
- Entity ids look like "path/to/file.py:Class.method" (files: "path/to/file.py").
- Start with search_entity. Use traverse_graph to follow calls, inheritance and imports. Use retrieve_entity to
  read code before you claim anything about it.
- Only cite entity ids that appeared in an observation. Never invent ids, line numbers or behaviour.
- If the evidence is insufficient, say so in the answer instead of guessing.
- You have at most {max_steps} tool calls. Answer as soon as you have enough evidence.
"""

MAX_OBS_CHARS = 6000


def _parse_action(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and "action" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
                break
        start = text.find("{", start + 1)
    return None


class ReActAgent:
    def __init__(self, llm: LLM, tools: dict, system_name: str, max_steps: int = 12,
                 max_tokens: int = 900):
        """tools: {tool_name: callable(**args) -> ToolResult}"""
        self.llm, self.tools, self.system_name = llm, tools, system_name
        self.max_steps, self.max_tokens = max_steps, max_tokens
        specs = [TOOL_SPECS[n] for n in tools]
        self.system = SYSTEM.format(
            max_steps=max_steps,
            tools="\n".join(f"- {s['name']}: {s['description']} args={json.dumps(s['args'])}" for s in specs))

    def run(self, question: str, repo: str) -> AgentAnswer:
        st = AgentState(question=question, repo=repo, max_steps=self.max_steps)
        st.messages.append({"role": "user", "content": f"REPOSITORY: {repo}\nQUESTION: {question}"})
        t0 = time.time()
        in_tok = out_tok = 0
        trace: list[dict] = []
        answer, cited = "", []
        for _ in range(self.max_steps + 1):
            force = st.step >= self.max_steps
            if force:
                st.budget_exhausted = True
                st.messages.append({"role": "user", "content":
                    "Step budget used up. Reply now with the finish action using what you have found."})
            r = self.llm.complete(self.system, st.messages, self.max_tokens, 0.0)
            in_tok, out_tok = in_tok + r.input_tokens, out_tok + r.output_tokens
            st.messages.append({"role": "assistant", "content": r.text})
            act = _parse_action(r.text)
            if act is None:
                if force:
                    answer = r.text.strip()
                    break
                st.step += 1
                st.messages.append({"role": "user", "content": "OBSERVATION: invalid reply. Send exactly one JSON "
                                    "object with keys thought, action, args."})
                trace.append({"step": st.step, "error": "invalid_json"})
                continue
            name, args = act["action"], act.get("args") or {}
            if name == "finish":
                answer = str(args.get("answer", "")).strip()
                cited = [e for e in (args.get("evidence") or []) if isinstance(e, str)]
                break
            if force:  # budget exhausted but the model keeps calling tools
                answer = "Budget exhausted before a final answer was produced."
                break
            st.step += 1
            res = self._call(name, args)
            st.tool_log.append(ToolCall(st.step, name, args, res.ok, len(res.text), res.entities))
            st.visited.update(res.entities)
            if name == "retrieve_entity" and res.ok:
                st.retrieved.update(res.data.get("retrieved", []))
            obs = res.text if len(res.text) <= MAX_OBS_CHARS else res.text[:MAX_OBS_CHARS] + "\n...(truncated)"
            st.messages.append({"role": "user", "content": f"OBSERVATION ({name}): {obs}"})
            trace.append({"step": st.step, "tool": name, "args": args, "ok": res.ok, "n_entities": len(res.entities)})
        grounded = [e for e in cited if e in st.visited]
        ungrounded = [e for e in cited if e not in st.visited]
        evidence = list(dict.fromkeys(sorted(st.retrieved) + grounded))
        if ungrounded:
            trace.append({"ungrounded_citations": ungrounded})
        return AgentAnswer(system=self.system_name, repo=repo, question=question, answer=answer,
                           evidence=evidence, tool_calls=len(st.tool_log), steps=st.step,
                           input_tokens=in_tok, output_tokens=out_tok, latency_s=time.time() - t0,
                           budget_exhausted=st.budget_exhausted, llm=self.llm.name, trace=trace)

    def _call(self, name: str, args: dict) -> ToolResult:
        fn = self.tools.get(name)
        if fn is None:
            return ToolResult(False, f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools)}")
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult(False, f"ERROR: bad arguments for {name}: {e}")
        except Exception as e:  # tools must never crash the loop
            return ToolResult(False, f"ERROR: {name} failed: {type(e).__name__}: {e}")
