"""Flat chunk retrieval for baseline B1 (non-agentic RAG).

Chunks are functions and classes (class chunks keep only the header and docstring when the class is long).
Retrieval is a hybrid of TF-IDF cosine (sparse vector) and BM25 rank fusion. Phase 1 planned Chroma dense
embeddings; see "Changes since Phase 1" in the README for why a local sparse retriever is used instead.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..graph.store import RepoGraph
from .entity_index import tokenize

MAX_CHUNK_LINES = 60


class ChunkIndex:
    def __init__(self, rg: RepoGraph):
        self.rg = rg
        self.ids, self.texts = [], []
        for nid, d in rg.entities(types=("class", "function")):
            code = rg.code(nid).splitlines()
            if d["type"] == "class" and len(code) > MAX_CHUNK_LINES:
                code = code[:15]
            self.ids.append(nid)
            self.texts.append(f"# {nid}\n" + "\n".join(code[:MAX_CHUNK_LINES]))
        docs = [tokenize(t) or ["_"] for t in self.texts]
        self.bm25 = BM25Okapi(docs)
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.mat = self.vec.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        bm = self.bm25.get_scores(toks)
        cos = linear_kernel(self.vec.transform([query]), self.mat).ravel()
        rank_bm = {i: r for r, i in enumerate(sorted(range(len(bm)), key=lambda i: -bm[i]))}
        rank_cos = {i: r for r, i in enumerate(sorted(range(len(cos)), key=lambda i: -cos[i]))}
        fused = {i: 1 / (60 + rank_bm[i]) + 1 / (60 + rank_cos[i]) for i in range(len(self.ids))}  # RRF
        best = sorted(fused, key=lambda i: -fused[i])[:top_k]
        return [(self.ids[i], fused[i]) for i in best]

    def text(self, nid: str) -> str:
        return self.texts[self.ids.index(nid)]
