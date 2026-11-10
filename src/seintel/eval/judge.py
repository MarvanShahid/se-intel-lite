"""LLM-as-a-judge with the official SWE-QA rubric (judge_prompt.txt is copied verbatim from
SWE-QA-Bench/Benchmark construction/score/llm-as-a-judge.py). Five dimensions scored 1-20; overall = sum (max 100).

One fixed judge model is used for every system so that scores are comparable (Phase 1 B.11).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..llm.base import LLM

PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text(encoding="utf-8")
DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


def parse_scores(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        s = {k: max(1, min(20, int(d[k]))) for k in DIMS}
    except (KeyError, TypeError, ValueError):
        return None
    s["overall"] = sum(s[k] for k in DIMS)
    return s


class Judge:
    def __init__(self, llm: LLM, retries: int = 2):
        self.llm, self.retries = llm, retries

    def score(self, question: str, reference: str, candidate: str) -> dict:
        if not candidate.strip():
            return {**{k: 1 for k in DIMS}, "overall": 5, "empty": True}
        prompt = PROMPT.replace("{question}", question).replace("{reference}", reference) \
                       .replace("{candidate}", candidate)
        msgs = [{"role": "user", "content": prompt}]
        for attempt in range(self.retries + 1):
            r = self.llm.complete("You are a strict evaluator. Output only JSON with integer SCORES.", msgs, 2000)
            s = parse_scores(r.text)
            if s:
                s["judge_tokens"] = r.input_tokens + r.output_tokens
                return s
            msgs = msgs[:1] + [{"role": "assistant", "content": r.text or "(empty)"},
                               {"role": "user", "content": "Output ONLY the JSON object with the 5 integer fields."}]
        return {"overall": None, "judge_error": True}
