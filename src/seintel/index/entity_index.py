"""Layered entity index used by SearchEntity (LocAgent [1]): exact id -> exact name -> BM25(ids) -> BM25(code)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..graph.store import RepoGraph

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: split on non-word chars, snake_case and CamelCase; lower-cased."""
    out = []
    for w in re.findall(r"[A-Za-z0-9_]+", text):
        for part in w.split("_"):
            if part:
                out.extend(p.lower() for p in _CAMEL.split(part) if p)
    return out


class EntityIndex:
    MAX_CODE_TOKENS = 400

    def __init__(self, rg: RepoGraph):
        self.rg = rg
        self.ids: list[str] = []
        self.by_name: dict[str, list[str]] = {}
        id_docs, code_docs = [], []
        for nid, d in rg.entities():
            self.ids.append(nid)
            self.by_name.setdefault(d["name"].lower(), []).append(nid)
            id_docs.append(tokenize(nid) or ["_"])
            code_docs.append((tokenize(rg.code(nid)) if d["type"] != "file" else tokenize(nid))[: self.MAX_CODE_TOKENS] or ["_"])
        self.bm25_id = BM25Okapi(id_docs)
        self.bm25_code = BM25Okapi(code_docs)

    def _ok(self, nid: str, types) -> bool:
        return types is None or self.rg.node(nid)["type"] in types

    def search(self, query: str, types=None, top_k: int = 8) -> list[tuple[str, str, float]]:
        """Return [(entity_id, layer, score)] with earlier layers ranked first."""
        q = query.strip()
        hits: list[tuple[str, str, float]] = []
        seen: set[str] = set()

        def add(nid, layer, score):
            if nid not in seen and self._ok(nid, types):
                seen.add(nid)
                hits.append((nid, layer, float(score)))

        if self.rg.has(q):
            add(q, "exact_id", 1e9)
        name = q.split(".")[-1].split(":")[-1].lower()
        for nid in sorted(self.by_name.get(name, []), key=len):
            add(nid, "exact_name", 1e8)
        toks = tokenize(q)
        if toks:
            for layer, bm in (("bm25_id", self.bm25_id), ("bm25_code", self.bm25_code)):
                if len(hits) >= top_k:
                    break
                scores = bm.get_scores(toks)
                for i in sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 3]:
                    if scores[i] <= 0 or len(hits) >= top_k:
                        break
                    add(self.ids[i], layer, scores[i])
        return hits[:top_k]
