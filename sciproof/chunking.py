"""Module 1: section-aware chunking.

Rules (each one exists for a reason, see the lesson):
  1. Never let a chunk cross a section boundary.
  2. Pack whole sentences, never cut mid-sentence.
  3. Overlap a little so facts at a boundary appear in two chunks.
  4. Skip sections that add noise (References, Acknowledgements).
  5. Attach metadata to every chunk so we can filter and cite later.
"""
from __future__ import annotations

import re

from .models import Chunk, Paper

SKIP_SECTIONS = {"references", "acknowledgements", "acknowledgments", "funding"}

# Abbreviations that end in a period but do NOT end a sentence.
_ABBREVIATIONS = ["et al.", "Fig.", "Figs.", "e.g.", "i.e.", "vs.", "approx.", "No."]
_PLACEHOLDER = "\u2024"  # one-dot leader: looks like '.', but our regex ignores it

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Naive but deliberate sentence splitter (see lesson for its limits)."""
    protected = text.strip()
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def chunk_section(
    sentences: list[str], max_chars: int = 600, overlap_sentences: int = 1
) -> list[str]:
    """Greedily pack sentences into chunks of at most ~max_chars.

    A single sentence longer than max_chars becomes its own chunk
    (we never split inside a sentence).
    """
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    chunks: list[list[str]] = []
    current: list[str] = []
    fresh = 0  # sentences in `current` that are new (not carried-over overlap)

    for sentence in sentences:
        candidate_len = len(" ".join(current + [sentence]))
        if current and candidate_len > max_chars:
            chunks.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            fresh = 0
            # if overlap + new sentence is still too long, drop the overlap
            if len(" ".join(current + [sentence])) > max_chars:
                current = []
        current.append(sentence)
        fresh += 1

    if current and fresh > 0:
        chunks.append(current)

    return [" ".join(c) for c in chunks]


def chunk_paper(
    paper: Paper, max_chars: int = 600, overlap_sentences: int = 1
) -> list[Chunk]:
    out: list[Chunk] = []
    for section in paper.sections:
        if section.name.strip().lower() in SKIP_SECTIONS:
            continue
        sentences = split_sentences(section.text)
        texts = chunk_section(sentences, max_chars, overlap_sentences)
        slug = section.name.strip().lower().replace(" ", "_")
        for i, text in enumerate(texts):
            out.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}::{slug}::{i}",
                    paper_id=paper.paper_id,
                    section=section.name,
                    position=i,
                    text=text,
                    year=paper.year,
                    study_type=paper.study_type,
                    title=paper.title,
                )
            )
    return out
