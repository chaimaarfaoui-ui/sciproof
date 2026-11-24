"""Modules 2-3: one facade over BM25, dense, hybrid (RRF) and hybrid + rerank."""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Chunk
from ..text import content_terms, entity_terms
from .bm25 import BM25Index
from .dense import DenseIndex
from .rerank import LexicalReranker

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float


def index_text(c: Chunk, use_title: bool) -> str:
    """Contextual indexing: prepend the paper title + section so a chunk that never
    names the compound (like most Results sections) is still findable."""
    return f"{c.title}. {c.section}. {c.text}" if use_title else c.text


class Retriever:
    def __init__(self, chunks: list[Chunk], mode: str = "hybrid_rerank",
                 use_title: bool = True, embedder=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.chunks, self.mode = chunks, mode
        docs = [index_text(c, use_title) for c in chunks]
        self.bm25 = BM25Index(docs)
        self.dense = DenseIndex(docs, embedder)
        self.reranker = LexicalReranker(self.bm25)
        self.idx = {c.chunk_id: i for i, c in enumerate(chunks)}

    def idf(self, term: str) -> float:
        return self.bm25.idf(term)

    def _rrf(self, *rankings, k=RRF_K):
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        wide = max(k * 4, 20)
        if self.mode == "bm25":
            ranked = self.bm25.search(query, k)
        elif self.mode == "dense":
            ranked = self.dense.search(query, k)
        else:
            fused = self._rrf(self.bm25.search(query, wide), self.dense.search(query, wide))
            if self.mode == "hybrid":
                ranked = fused[:k]
            else:
                cand = fused[:wide]
                ranked = sorted(
                    ((i, self.reranker.score(query, i, self.chunks[i].section)) for i, _ in cand),
                    key=lambda x: -x[1],
                )[:k]
        return [Hit(self.chunks[i], s) for i, s in ranked]

    def paper_coverage(self, query: str, hits: list[Hit]) -> dict:
        """paper_id -> (idf-weighted share of the query's topic terms that paper covers,
        missing terms). A question about X-on-Y needs ONE paper that mentions both."""
        terms = set(content_terms(query))
        if not terms:
            return {}
        by_paper: dict[str, set] = {}
        for h in hits:
            by_paper.setdefault(h.chunk.paper_id, set()).update(self.bm25.tokens[self.idx[h.chunk.chunk_id]])
        ents = entity_terms(query) & terms
        total = sum(self.idf(t) for t in terms)
        out = {}
        for pid, have in by_paper.items():
            cov = sum(self.idf(t) for t in terms if t in have) / total
            if ents - have:          # a named entity is missing: a weighted average can't excuse that
                cov = min(cov, 0.5)
            out[pid] = (cov, sorted(t for t in terms if t not in have))
        return out

    def coverage(self, query: str, hits: list[Hit]):
        """Best single-paper coverage: our 'do we even have the evidence?' test."""
        pc = self.paper_coverage(query, hits)
        if not pc:
            return 0.0, []
        pid = max(pc, key=lambda p: pc[p][0])
        return pc[pid]
