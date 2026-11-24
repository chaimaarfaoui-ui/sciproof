"""Module 2: tokenizing and small text utilities shared by every component."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NEGATION = re.compile(r"\b(no|not|without|never|neither|nor|failed|fail|cannot)\b|n't")

STOPWORDS = set(
    "a an the of on in to for and or is are was were be been being what which who whom how "
    "does do did with by from at as that this it its than then there their can could would "
    "should about between into over after before during versus vs p we our us they them".split()
)
# Words that make a *question* a question but say nothing about the topic.
QUESTION_WORDS = set(
    "effect effects evidence consistent consistently change changed much many show shown "
    "lower lowers lowering reduce reduces reduced increase increases raise raises improve improves "
    "affect affects cause causes modify work works "
    "finding findings study studies result results across compare compared".split()
)


def stem(tok: str) -> str:
    """Very light suffix stripper: lowers/lowered/lowering -> lower."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            tok = tok[: -len(suf)]
            break
    if tok.endswith("e") and len(tok) > 3:
        tok = tok[:-1]
    return tok


def tokenize(text: str, drop_question_words: bool = False) -> list[str]:
    out = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        if drop_question_words and raw in QUESTION_WORDS:
            continue
        if len(raw) < 2 and not raw.isdigit():
            continue
        out.append(stem(raw))
    return out


def content_terms(text: str) -> list[str]:
    """Topic-bearing terms of a question or claim."""
    return tokenize(text, drop_question_words=True)


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(text.lower()))


_ENTITY = re.compile(r"\b[A-Za-z]{1,6}-?\d+[A-Za-z0-9-]*\b|\b[A-Z]{2,}\b")


def entity_terms(text: str) -> set[str]:
    """Identifier-like tokens (ZX-101, IL-6, BRCA1, HbA1c, ECG) that must match exactly.

    Heuristic stand-in for biomedical named-entity recognition (e.g. scispaCy)."""
    out: set[str] = set()
    for m in _ENTITY.findall(text):
        out.update(tokenize(m))
    return out
