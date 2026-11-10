"""Non-agentic baselines.

B0 plain LLM: the same LLM answers with no repository context.
B1 flat RAG : top-k function/class chunks (hybrid BM25 + TF-IDF, rank fusion) are put in one prompt.
"""
from __future__ import annotations

import time

from ..agent.state import AgentAnswer
from ..index.chunk_index import ChunkIndex
from ..llm.base import LLM

B0_SYSTEM = ("You are an expert on the Python repository named by the user. Answer the question about it as "
             "precisely as you can, naming the relevant files, classes and functions.")
B1_SYSTEM = ("You answer questions about a Python repository using ONLY the retrieved code chunks below. Name "
             "the relevant files, classes and functions (as path/file.py:Name). If the chunks are not enough, "
             "say what is missing instead of guessing.")


class PlainLLM:
    def __init__(self, llm: LLM, max_tokens: int = 16000):
        self.llm, self.max_tokens = llm, max_tokens

    def run(self, question: str, repo: str) -> AgentAnswer:
        t0 = time.time()
        r = self.llm.complete(B0_SYSTEM, [{"role": "user", "content": f"REPOSITORY: {repo}\nQUESTION: {question}"}],
                              self.max_tokens)
        return AgentAnswer("B0", repo, question, r.text.strip(), [], 0, 1, r.input_tokens, r.output_tokens,
                           time.time() - t0, False, self.llm.name, cache_read=r.cache_read,
                           cache_write=r.cache_write)


class FlatRAG:
    def __init__(self, llm: LLM, index: ChunkIndex, top_k: int = 8, max_tokens: int = 16000):
        self.llm, self.index, self.top_k, self.max_tokens = llm, index, top_k, max_tokens

    def run(self, question: str, repo: str) -> AgentAnswer:
        t0 = time.time()
        hits = self.index.search(question, self.top_k)
        ctx = "\n\n".join(self.index.text(nid) for nid, _ in hits)
        r = self.llm.complete(B1_SYSTEM, [{"role": "user", "content":
                                           f"REPOSITORY: {repo}\nRETRIEVED CHUNKS:\n{ctx}\n\nQUESTION: {question}"}],
                              self.max_tokens)
        return AgentAnswer("B1", repo, question, r.text.strip(), [h[0] for h in hits], 0, 1, r.input_tokens,
                           r.output_tokens, time.time() - t0, False, self.llm.name,
                           trace=[{"retrieved": [h[0] for h in hits]}], cache_read=r.cache_read,
                           cache_write=r.cache_write)
