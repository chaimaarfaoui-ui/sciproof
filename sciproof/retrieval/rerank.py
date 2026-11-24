"""Module 3: a transparent reranker.

First-stage retrieval is cheap and broad; the reranker spends more effort on the few
candidates that survive. This one is lexical and hand-weighted so you can read it. In
production this slot usually holds a cross-encoder model.
"""
from __future__ import annotations

from ..text import content_terms

SECTION_PRIOR = {"results": 1.0, "abstract": 0.95, "discussion": 0.8, "methods": 0.6}


class LexicalReranker:
    W_COVERAGE, W_BIGRAM, W_SECTION = 0.6, 0.3, 0.1

    def __init__(self, bm25):
        self.bm25 = bm25

    def score(self, query: str, doc_idx: int, section: str) -> float:
        q = content_terms(query)
        if not q:
            return 0.0
        toks = self.bm25.tokens[doc_idx]
        have = set(toks)
        qset = set(q)
        total = sum(self.bm25.idf(t) for t in qset) or 1.0
        coverage = sum(self.bm25.idf(t) for t in qset if t in have) / total
        bigrams = list(zip(q, q[1:]))
        doc_bi = set(zip(toks, toks[1:]))
        bigram = (sum(1 for b in bigrams if b in doc_bi) / len(bigrams)) if bigrams else 0.0
        prior = SECTION_PRIOR.get(section.lower(), 0.7)
        return self.W_COVERAGE * coverage + self.W_BIGRAM * bigram + self.W_SECTION * prior
