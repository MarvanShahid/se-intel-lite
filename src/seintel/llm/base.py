"""Provider-neutral LLM interface, disk cache, pricing and a factory.

Every system (B0, B1, B2, P) and the judge talk to an `LLM`. Responses are cached on disk keyed by
(model, system, messages, max_tokens, seed/run id) so that re-running an experiment is free and exactly
reproduces the saved traces; a different `run` id gives an independent sample for repeated runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# USD per 1M tokens (input, output). Anthropic first-party list prices, cached 2026-10-06.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "mock": (0.0, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    """input_tokens = uncached input. Cache reads bill at 0.1x and 5-minute cache writes at 1.25x input price."""
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * pin + cache_read * pin * 0.1 + cache_write * pin * 1.25 + output_tokens * pout) / 1e6


class MissingCredentials(RuntimeError):
    """No API credentials: experiments must stop instead of logging the same error for every question."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0
    stop_reason: str = ""
    cache_read: int = 0
    cache_write: int = 0


class LLM:
    """Base class. Subclasses implement `_complete`."""
    name = "base"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 16000,
                 temperature: float | None = None) -> LLMResponse:
        t0 = time.time()
        r = self._complete(system, messages, max_tokens)
        r.latency_s = r.latency_s or (time.time() - t0)
        return r

    def _complete(self, system: str, messages: list[dict], max_tokens: int) -> LLMResponse:
        raise NotImplementedError


class CachedLLM(LLM):
    """Disk cache wrapper. `run` separates independent repeated runs (run 0, 1, 2 ...)."""

    def __init__(self, inner: LLM, cache_dir: str | Path, run: int = 0):
        self.inner, self.run = inner, run
        self.name = inner.name
        self.dir = Path(cache_dir) / inner.name.replace("/", "_")
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, system, messages, max_tokens) -> str:
        blob = json.dumps({"s": system, "m": messages, "t": max_tokens, "run": self.run},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system, messages, max_tokens=16000, temperature=None) -> LLMResponse:
        k = self._key(system, messages, max_tokens)
        f = self.dir / k[:2] / f"{k}.json"
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                return LLMResponse(**{**d, "cached": True})
            except (json.JSONDecodeError, TypeError):
                pass
        r = self.inner.complete(system, messages, max_tokens)
        if r.stop_reason == "error":     # never cache failures, or a rerun would replay them
            return r
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(asdict(r), ensure_ascii=False), encoding="utf-8")
        return r


def make_llm(spec: str, cache_dir: str | Path | None = ".cache/llm", run: int = 0, **kw) -> LLM:
    """spec examples: 'anthropic:claude-opus-5-5', 'openai:gpt-x' (any OpenAI-compatible endpoint), 'mock'."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        llm: LLM = AnthropicLLM(model or os.environ.get("SEINTEL_MODEL", "claude-opus-5-5"), **kw)
    elif provider == "openai":
        from .openai_llm import OpenAILLM
        llm = OpenAILLM(model, **kw)
    elif provider == "mock":
        from .mock import MockLLM
        llm = MockLLM(**kw)
    else:
        raise ValueError(f"unknown LLM provider '{provider}' (use anthropic:<model>, openai:<model> or mock)")
    return CachedLLM(llm, cache_dir, run) if cache_dir and provider != "mock" else llm
