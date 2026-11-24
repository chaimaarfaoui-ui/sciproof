"""Module 5: the verification pass ("trust, but check").

A generator can cite a real passage and still say something the passage does not say.
This checker re-reads each claim against the passages it cites and fails it if:
  - it cites a passage we never retrieved           (fabricated citation)
  - too few of its content terms appear in the cited text   (unsupported wording)
  - it contains a number the cited text does not contain    (fabricated statistic)
  - its negation differs from the best-matching sentence    (flipped meaning)

HONEST LIMIT: this is lexical, not true entailment. It catches the common, expensive
failures (made-up numbers, wrong citations, flipped negation) cheaply and
deterministically. Paraphrased contradictions can slip through; for those, plug an NLI
model or LLM judge into the same `verify()` interface.
"""
from __future__ import annotations

from dataclasses import replace

from .models import Claim
from .splitting import split_sentences
from .text import content_terms, has_negation, numbers


class LexicalVerifier:
    def __init__(self, min_overlap: float = 0.8):
        self.min_overlap = min_overlap

    def verify(self, claim: Claim, passages: dict) -> Claim:
        if not claim.chunk_ids:
            return replace(claim, verified=False, note="no citation")
        unknown = [c for c in claim.chunk_ids if c not in passages]
        if unknown:
            return replace(claim, verified=False, note=f"cites unknown passage {unknown[0]}")
        cited_text = " ".join(passages[c].text for c in claim.chunk_ids)
        cterms = set(content_terms(claim.text))
        have = set(content_terms(cited_text))
        overlap = len(cterms & have) / len(cterms) if cterms else 0.0
        if overlap < self.min_overlap:
            return replace(claim, verified=False, note=f"low textual support ({overlap:.0%})")
        missing_nums = numbers(claim.text) - numbers(cited_text)
        if missing_nums:
            return replace(claim, verified=False, note=f"number(s) not in source: {sorted(missing_nums)}")
        sents = split_sentences(cited_text)
        best = max(sents, key=lambda s: len(cterms & set(content_terms(s))), default="")
        if has_negation(claim.text) != has_negation(best):
            return replace(claim, verified=False, note="negation differs from source sentence")
        return replace(claim, verified=True, note=f"supported ({overlap:.0%} term overlap)")
