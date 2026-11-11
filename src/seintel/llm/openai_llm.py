"""Provider for any OpenAI-compatible endpoint: Google Gemini (AI Studio), Groq, OpenRouter, a local Ollama server, ...

Set OPENAI_BASE_URL and OPENAI_API_KEY, then use --llm openai:<model>. Examples (see README):
  Gemini     OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  Groq       OPENAI_BASE_URL=https://api.groq.com/openai/v1
  OpenRouter OPENAI_BASE_URL=https://openrouter.ai/api/v1
  Ollama     OPENAI_BASE_URL=http://localhost:11434/v1   (OPENAI_API_KEY=ollama)

Free tiers are rate limited, so 429 responses are retried with exponential backoff (up to ~10 minutes in total)
instead of failing the question. SEINTEL_MAX_TOKENS caps the output length (some providers reject 16000).
"""
from __future__ import annotations

import os
import random
import time

from .base import LLM, LLMResponse, MissingCredentials


class OpenAILLM(LLM):
    def __init__(self, model: str, max_wait_s: float = 600.0, **_):
        import openai
        if not model:
            raise ValueError("use --llm openai:<model name>, e.g. openai:gemini-2.5-flash")
        self._openai = openai
        self.model = self.name = model
        self.max_wait_s = max_wait_s
        self.cap = int(os.environ.get("SEINTEL_MAX_TOKENS", "8192"))
        self.min_interval = float(os.environ.get("SEINTEL_MIN_INTERVAL_S", "0"))   # e.g. 4 for a 15 RPM free tier
        self._last = 0.0
        try:
            self.client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None, max_retries=0)
        except openai.OpenAIError as e:   # raised when OPENAI_API_KEY is missing
            raise MissingCredentials(f"No OpenAI-compatible API key found ({e}). Set OPENAI_API_KEY and "
                                     "OPENAI_BASE_URL for your provider.") from None

    def _complete(self, system, messages, max_tokens) -> LLMResponse:
        oa = self._openai
        t0, delay = time.time(), 5.0
        while True:
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=min(max_tokens, self.cap),
                    messages=[{"role": "system", "content": system}, *messages])
                break
            except oa.AuthenticationError:
                raise MissingCredentials("The API key was rejected by the provider at "
                                         f"{os.environ.get('OPENAI_BASE_URL', 'api.openai.com')}.") from None
            except oa.NotFoundError as e:
                raise MissingCredentials(f"Model '{self.model}' not found at this provider: {e}") from None
            except (oa.RateLimitError, oa.APIConnectionError, oa.InternalServerError, oa.APITimeoutError) as e:
                if time.time() - t0 > self.max_wait_s:
                    return LLMResponse(text=f"LLM_ERROR: gave up after retries: {type(e).__name__}",
                                       stop_reason="error", latency_s=time.time() - t0)
                time.sleep(delay + random.uniform(0, 1))
                delay = min(delay * 2, 60.0)
            except oa.BadRequestError as e:
                return LLMResponse(text=f"LLM_ERROR: bad request: {str(e)[:300]}", stop_reason="error",
                                   latency_s=time.time() - t0)
        u = r.usage
        choice = r.choices[0] if r.choices else None
        return LLMResponse(text=(choice.message.content if choice else "") or "",
                           input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                           output_tokens=getattr(u, "completion_tokens", 0) or 0,
                           latency_s=time.time() - t0, stop_reason=(choice.finish_reason if choice else "") or "")
