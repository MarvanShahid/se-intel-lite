"""ReAct style agent with a step budget, typed tools and an evidence trail.

The same loop is used for baseline B2 (3 graph tools, LocAgent-style protocol) and the proposed system P
(+ RunQuality, GetHotspots, GetRisk, the QA protocol and the citation verifier).

Gap 1 components (switchable for ablation):
- qa_prompt: question-answering protocol (identify named entities, follow typed edges for multi-hop questions,
  answer with mechanism + cited entity ids) instead of the plain LocAgent-style "explore then answer" prompt.
- verify_citations: before accepting `finish`, check that every cited id was surfaced by a tool and that code was
  actually read; otherwise return a VERIFICATION observation and allow one revision.
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
{extra}"""

QA_PROTOCOL = """
QUESTION ANSWERING PROTOCOL
- First identify the entities the question names (functions, classes, options, files) and search for each.
- For "how/why" and multi-hop questions, follow typed edges: invokes (what it calls), invoked_by (who calls it),
  inherits/inherited_by, imports/imported_by. Read the key functions on the path with retrieve_entity.
- For questions about code quality, complexity, maintainability, hotspots or fragility use run_quality or
  get_hotspots; for questions about risk or the safety of changing code use get_risk. Report the numbers you got.
- The answer must: (1) answer the question directly in the first sentence, (2) explain the mechanism step by step,
  referring to concrete functions/classes as `path/file.py:Name`, (3) state what is uncertain.
- evidence must list every entity id the answer relies on.
"""

VERIFY_MSG = ("VERIFICATION: your answer was not accepted yet. {problems} Fix this: call retrieve_entity or "
              "search_entity if needed, then call finish again citing only ids that appeared in observations.")

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
                 max_tokens: int = 16000, qa_prompt: bool = False, verify_citations: bool = False,
                 max_revisions: int = 1):
        """tools: {tool_name: callable(**args) -> ToolResult}"""
        self.llm, self.tools, self.system_name = llm, tools, system_name
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.verify_citations, self.max_revisions = verify_citations, max_revisions
        specs = [TOOL_SPECS[n] for n in tools]
        self.system = SYSTEM.format(
            max_steps=max_steps, extra=QA_PROTOCOL if qa_prompt else "",
            tools="\n".join(f"- {s['name']}: {s['description']} args={json.dumps(s['args'])}" for s in specs))

    def _verify(self, st: AgentState, answer: str, cited: list[str]) -> str | None:
        """Describe grounding problems of a candidate answer, or None if it is acceptable."""
        problems = []
        bad = [e for e in cited if e not in st.visited]
        if bad:
            problems.append(f"These cited ids never appeared in an observation: {', '.join(bad[:6])}.")
        if not cited:
            problems.append("The evidence list is empty.")
        if not st.retrieved and "retrieve_entity" in self.tools:
            problems.append("You have not read any code with retrieve_entity.")
        if len(answer.strip()) < 40:
            problems.append("The answer is too short to explain the mechanism.")
        return " ".join(problems) or None

    def run(self, question: str, repo: str) -> AgentAnswer:
        st = AgentState(question=question, repo=repo, max_steps=self.max_steps)
        st.messages.append({"role": "user", "content": f"REPOSITORY: {repo}\nQUESTION: {question}"})
        t0 = time.time()
        in_tok = out_tok = c_read = c_write = 0
        trace: list[dict] = []
        answer, cited = "", []
        revisions = llm_errors = 0
        for _ in range(self.max_steps + self.max_revisions + 4):
            force = st.step >= self.max_steps
            if force and not st.budget_exhausted:
                st.budget_exhausted = True
                st.messages.append({"role": "user", "content":
                    "Step budget used up. Reply now with the finish action using what you have found."})
            r = self.llm.complete(self.system, st.messages, self.max_tokens)
            in_tok, out_tok = in_tok + r.input_tokens, out_tok + r.output_tokens
            c_read, c_write = c_read + r.cache_read, c_write + r.cache_write
            if r.stop_reason == "error" or r.text.startswith("LLM_ERROR"):
                llm_errors += 1
                trace.append({"step": st.step, "error": "llm_error", "detail": r.text[:300]})
                if llm_errors >= 3:
                    answer = "The language model failed repeatedly; no answer was produced."
                    break
                continue
            st.messages.append({"role": "assistant", "content": r.text or "(empty)"})
            act = _parse_action(r.text)
            if act is None:
                if force:
                    answer = r.text.strip()
                    break
                st.step += 1
                st.messages.append({"role": "user", "content": "OBSERVATION: invalid reply. Send exactly one JSON "
                                    "object with keys thought, action, args."})
                trace.append({"step": st.step, "error": "invalid_json", "raw": r.text[:200]})
                continue
            name, args = act["action"], act.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if name == "finish":
                answer = str(args.get("answer", "")).strip()
                cited = [e for e in (args.get("evidence") or []) if isinstance(e, str)]
                problem = self._verify(st, answer, cited) if self.verify_citations else None
                if problem and revisions < self.max_revisions and not force:
                    revisions += 1
                    trace.append({"step": st.step, "verification_failed": problem, "thought": act.get("thought", "")})
                    st.messages.append({"role": "user", "content": VERIFY_MSG.format(problems=problem)})
                    continue
                trace.append({"step": st.step, "action": "finish", "thought": act.get("thought", ""),
                              "n_cited": len(cited),
                              "verification": "off" if not self.verify_citations else
                              ("passed" if not problem else "failed_final")})
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
            trace.append({"step": st.step, "thought": act.get("thought", ""), "tool": name, "args": args,
                          "ok": res.ok, "n_entities": len(res.entities), "obs_preview": res.text[:300],
                          "state": {"visited": len(st.visited), "retrieved": len(st.retrieved)}})
        grounded = [e for e in cited if e in st.visited]
        ungrounded = [e for e in cited if e not in st.visited]
        evidence = list(dict.fromkeys(sorted(st.retrieved) + grounded))
        if ungrounded:
            trace.append({"ungrounded_citations": ungrounded})
        return AgentAnswer(system=self.system_name, repo=repo, question=question, answer=answer,
                           evidence=evidence, tool_calls=len(st.tool_log), steps=st.step,
                           input_tokens=in_tok, output_tokens=out_tok, latency_s=time.time() - t0,
                           budget_exhausted=st.budget_exhausted, llm=self.llm.name, trace=trace,
                           revisions=revisions, cited=len(cited), ungrounded=len(ungrounded),
                           cache_read=c_read, cache_write=c_write)

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
