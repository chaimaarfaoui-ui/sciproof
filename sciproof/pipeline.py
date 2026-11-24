"""Modules 4-6: the end-to-end pipeline that ties everything together.

retrieve -> (abstain?) -> generate with citations -> verify each claim -> assess evidence
"""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_paper
from .evidence import assess
from .generation import ExtractiveGenerator, GenerationError
from .loader import load_papers
from .models import Answer, Assessment
from .retrieval import Retriever
from .verification import LexicalVerifier

COVERAGE_THRESHOLD = 0.8


class Sciproof:
    def __init__(self, papers, mode="hybrid_rerank", generator=None, verifier=None,
                 max_chars=600, overlap=1, use_title=True, embedder=None, k=8):
        self.papers = {p.paper_id: p for p in papers}
        self.chunks = [c for p in papers for c in chunk_paper(
            p, max_chars, overlap)]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.retriever = Retriever(self.chunks, mode, use_title, embedder)
        self.generator = generator or ExtractiveGenerator()
        self.fallback = ExtractiveGenerator()
        self.verifier = verifier or LexicalVerifier()
        self.k, self.mode = k, mode

    @classmethod
    def from_file(cls, path, **kw):
        return cls(load_papers(Path(path)), **kw)

    def _relevant_hits(self, text: str):
        """Retrieve wide, then keep only papers that cover the WHOLE question."""
        wide = self.retriever.search(text, 12)
        pc = self.retriever.paper_coverage(text, wide)
        best = max((v[0] for v in pc.values()), default=0.0)
        missing = pc[max(pc, key=lambda p: pc[p][0])][1] if pc else []
        kept = [h for h in wide if pc.get(h.chunk.paper_id, (0,))[
            0] >= COVERAGE_THRESHOLD]
        return kept[: self.k], best, missing

    def ask(self, question: str) -> Answer:
        hits, cov, missing = self._relevant_hits(question)
        if not hits:
            why = (f"The corpus does not appear to cover: {', '.join(missing)}." if missing
                   else "No relevant passages found.")
            return Answer(question, [], True, why, {}, None, [], self.mode)

        notes = []
        try:
            claims = self.generator.generate(
                question, hits, self.retriever.idf)
        except GenerationError as e:
            notes.append(f"generator failed ({e}); fell back to extractive")
            claims = self.fallback.generate(question, hits, self.retriever.idf)

        retrieved = {h.chunk.chunk_id: h.chunk for h in hits}
        checked = [self.verifier.verify(c, retrieved) for c in claims]
        good = [c for c in checked if c.verified]
        for c in checked:
            if not c.verified:
                notes.append(
                    f"dropped unverified claim: {c.text[:60]}... ({c.note})")
        if not good:
            return Answer(question, [], True, "No claim could be verified against the sources.", {}, None, notes, self.mode)

        cited = {cid: retrieved[cid] for c in good for cid in c.chunk_ids}
        return Answer(question, good, False, "", cited, assess(question, hits, self.retriever.idf), notes, self.mode)

    def check_claim(self, claim: str):
        """Fact-check a statement: what does the corpus say about it?"""
        hits, cov, missing = self._relevant_hits(claim)
        if not hits:
            return Assessment("insufficient", f"The corpus does not appear to cover: {', '.join(missing)}.", [])
        return assess(claim, hits, self.retriever.idf)


def answer_to_dict(a: Answer) -> dict:
    def study(s):
        return dict(paper_id=s.paper_id, title=s.title, study_type=s.study_type, year=s.year,
                    stance=s.stance, chunk_id=s.chunk_id, sentence=s.sentence)
    return {
        "question": a.question, "abstained": a.abstained, "reason": a.reason, "mode": a.retrieval_mode,
        "claims": [dict(text=c.text, chunk_ids=list(c.chunk_ids), verified=c.verified, note=c.note) for c in a.claims],
        "passages": {k: dict(chunk_id=v.chunk_id, paper_id=v.paper_id, title=v.title, section=v.section,
                             study_type=v.study_type, year=v.year, text=v.text) for k, v in a.passages.items()},
        "assessment": None if not a.assessment else dict(
            verdict=a.assessment.verdict, explanation=a.assessment.explanation,
            studies=[study(s) for s in a.assessment.studies]),
        "notes": a.notes,
    }
