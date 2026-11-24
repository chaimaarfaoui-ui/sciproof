"""Module 1: the data model.

Everything downstream (retrieval, citations, evaluation) depends on these
shapes, so we define them first and keep them small and explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    name: str   # e.g. "Abstract", "Methods", "Results"
    text: str


@dataclass
class Paper:
    paper_id: str
    title: str
    year: int
    study_type: str  # e.g. "RCT", "cohort", "review"
    sections: list[Section] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    """The unit we index, retrieve, and cite.

    chunk_id is stable and human-readable, so a citation can always be
    traced back: "<paper_id>::<section>::<n>".
    """
    chunk_id: str
    paper_id: str
    section: str
    position: int      # index of this chunk inside its section
    text: str
    year: int
    study_type: str
    title: str = ""   # paper title, used as retrieval context (Module 2)


@dataclass(frozen=True)
class Claim:
    """One sentence of an answer plus the passages that back it."""
    text: str
    chunk_ids: tuple = ()
    verified: bool | None = None   # set by the verifier
    note: str = ""                 # why it passed or failed


@dataclass
class StudyStance:
    paper_id: str
    title: str
    study_type: str
    year: int
    stance: str        # "effect" | "null" | "unclear"
    chunk_id: str
    sentence: str


@dataclass
class Assessment:
    verdict: str       # supported | not_supported | conflicting | insufficient
    explanation: str
    studies: list = field(default_factory=list)


@dataclass
class Answer:
    question: str
    claims: list
    abstained: bool
    reason: str
    passages: dict          # chunk_id -> Chunk (everything a claim cites)
    assessment: Assessment | None = None
    notes: list = field(default_factory=list)
    retrieval_mode: str = ""
