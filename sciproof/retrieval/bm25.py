"""Module 2: BM25 from scratch (so you can see every term in the formula)."""
from __future__ import annotations

import math
from collections import Counter

from ..text import tokenize


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.N) if self.N else 1.0
        self.tf = [Counter(t) for t in self.tokens]
        self.df: Counter = Counter()
        for toks in self.tokens:
            self.df.update(set(toks))

    def idf(self, term: str) -> float:
        """Rare terms score high. Unknown (df=0) terms get the maximum idf."""
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            dl = len(self.tokens[i])
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(self.N), key=lambda i: -sc[i])
        return [(i, sc[i]) for i in order[:k] if sc[i] > 0]
