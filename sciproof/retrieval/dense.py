"""Module 3: dense (semantic-vector) retrieval behind a swappable interface.

Default embedder = LSA (TF-IDF + SVD) built with numpy: it needs no download and no
GPU, and it is a *real* dense retriever (it can match documents that share no exact
words with the query, via co-occurrence). In production you would swap in a neural
embedding model, e.g. SentenceTransformerEmbedder below.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..text import tokenize


class LSAEmbedder:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def _tfidf(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, t in enumerate(texts):
            for term, f in Counter(tokenize(t)).items():
                j = self.vocab.get(term)
                if j is not None:
                    X[i, j] = (1 + math.log(f)) * self.idf[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def fit(self, texts: list[str]):
        toks = [set(tokenize(t)) for t in texts]
        terms = sorted(set().union(*toks))
        self.vocab = {t: i for i, t in enumerate(terms)}
        df = np.zeros(len(terms))
        for s in toks:
            for t in s:
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)) + 1
        X = self._tfidf(texts)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.V = Vt[: min(self.dim, Vt.shape[0])].T  # vocab x k
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        Z = self._tfidf(texts) @ self.V
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.where(n == 0, 1, n)


class SentenceTransformerEmbedder:
    """Optional neural embedder. Needs `pip install sentence-transformers` + internet
    to download weights, so it is NOT exercised by the offline tests."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class DenseIndex:
    def __init__(self, docs: list[str], embedder=None):
        self.embedder = (embedder or LSAEmbedder()).fit(docs)
        self.D = self.embedder.encode(docs)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.D @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]
