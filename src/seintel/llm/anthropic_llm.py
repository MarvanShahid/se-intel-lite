"""Claude via the official Anthropic SDK.

Notes for the current model family (Opus 5.5 / Sonnet 5.5 / Haiku 5.5):
- sampling parameters (temperature/top_p) are not accepted, so runs are stochastic; we repeat each
  configuration 3 times and report mean and standard deviation;
- thinking is adaptive and cannot be disabled on Opus 5.5, so depth is controlled with `effort`;
- the response may contain thinking blocks; only text blocks are returned to the agent.
"""
from __future__ import annotations

import os
import time

import anthropic

from .base import LLM, LLMResponse, MissingCredentials


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = None, max_retries: int = 5):
        self.model = model
        self.name = model
        self.effort = effort or os.environ.get("SEINTEL_EFFORT", "medium")
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def _complete(self, system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        t0 = time.time()
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},   # automatic prefix caching of the growing agent transcript
            )
        except TypeError as e:
            if "authentication" in str(e).lower():
                raise MissingCredentials(
                    "No Anthropic API key found. Set it in this terminal first, e.g. PowerShell: "
                    '$env:ANTHROPIC_API_KEY = "sk-ant-..."  (or run `ant auth login`).') from None
            raise
        except anthropic.AuthenticationError:
            raise MissingCredentials("The Anthropic API key was rejected (invalid or revoked). "
                                     "Create a new key at https://platform.claude.com.") from None
        except anthropic.BadRequestError as e:
            # surfaced to the agent loop, which records it in the trace instead of crashing the run
            return LLMResponse(text=f"LLM_ERROR: bad request: {e.message}", stop_reason="error",
                               latency_s=time.time() - t0)
        except anthropic.APIConnectionError as e:
            return LLMResponse(text=f"LLM_ERROR: connection: {e}", stop_reason="error",
                               latency_s=time.time() - t0)
        text = "".join(b.text for b in r.content if b.type == "text")
        if r.stop_reason == "refusal":
            text = text or "LLM_ERROR: refusal"
        u = r.usage
        return LLMResponse(text=text, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                           latency_s=time.time() - t0, stop_reason=r.stop_reason or "",
                           cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                           cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
