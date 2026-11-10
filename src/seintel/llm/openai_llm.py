"""Optional provider for any OpenAI-compatible endpoint (kept from Phase 1; set OPENAI_BASE_URL for local
servers such as Ollama or vLLM). Not used for the reported results."""
from __future__ import annotations

import os
import time

from .base import LLM, LLMResponse


class OpenAILLM(LLM):
    def __init__(self, model: str, **_):
        from openai import OpenAI
        self.model = self.name = model
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    def _complete(self, system, messages, max_tokens) -> LLMResponse:
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages])
        u = r.usage
        return LLMResponse(text=r.choices[0].message.content or "",
                           input_tokens=getattr(u, "prompt_tokens", 0), output_tokens=getattr(u, "completion_tokens", 0),
                           latency_s=time.time() - t0, stop_reason=r.choices[0].finish_reason or "")
