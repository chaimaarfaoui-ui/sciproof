"""Module 5: weighing evidence across studies.

Two jobs: (1) decide each study's stance toward a claim, (2) combine stances into a
verdict that respects study design (an RCT outweighs a cohort for causal claims).

HONEST LIMIT: stance detection here is rule-based. It detects 'reports a significant
effect' vs 'reports no effect' (via p-values and cue phrases). It does NOT verify the
DIRECTION of the effect or handle subgroup nuance. Replace stance_of() with an NLI model
or an LLM judge for real use; the interface stays the same.
"""
from __future__ import annotations

import re

from .models import Assessment, StudyStance
from .splitting import split_sentences
from .text import content_terms

DESIGN_RANK = {"meta-analysis": 5, "RCT": 4, "cohort": 3, "case-control": 2, "review": 1}
PRIMARY = {"RCT", "cohort", "case-control", "meta-analysis"}  # reviews are secondary

_P = re.compile(r"\bp\s*([=<>])\s*(\d*\.\d+)")
_NULL = ("no meaningful", "not significant", "no significant", "no change", "no difference",
         "no effect", "not statistically", "did not")
_EFFECT = ("fell", "dropped", "decreased", "reduced", "lower", "improved", "declined", "increased")


def stance_of(sentence: str) -> str:
    s = sentence.lower()
    if any(c in s for c in _NULL):
        return "null"
    m = _P.search(s)
    if m:
        op, val = m.group(1), float(m.group(2))
        if op == "=" and val >= 0.05:
            return "null"
        if val < 0.05 or (op == "<" and val <= 0.05):
            return "effect"
    if any(c in s for c in _EFFECT):
        return "effect"
    return "unclear"


def assess(query: str, hits, idf) -> Assessment:
    qterms = set(content_terms(query))
    denom = sum(idf(t) for t in qterms) or 1.0
    best: dict[str, tuple] = {}
    for h in hits:
        c = h.chunk
        if c.study_type not in PRIMARY or c.section.lower() not in ("abstract", "results"):
            continue
        for sent in split_sentences(c.text):
            ov = sum(idf(t) for t in qterms & set(content_terms(sent))) / denom
            # a sentence with a p-value or an explicit null cue carries a verdict
            has_signal = stance_of(sent) != "unclear"
            score = ov + (0.5 if has_signal else 0.0) + (0.1 if c.section.lower() == "results" else 0.0)
            if c.paper_id not in best or score > best[c.paper_id][0]:
                best[c.paper_id] = (score, c, sent)
    studies = [StudyStance(c.paper_id, c.title, c.study_type, c.year, stance_of(s), c.chunk_id, s)
               for _, c, s in best.values()]
    studies.sort(key=lambda x: -DESIGN_RANK.get(x.study_type, 0))
    eff = [s for s in studies if s.stance == "effect"]
    nul = [s for s in studies if s.stance == "null"]

    def desc(xs):
        return ", ".join(f"{x.study_type} {x.year}" for x in xs)

    if not eff and not nul:
        return Assessment("insufficient", "No primary study in the retrieved evidence reports a clear result.", studies)
    if eff and not nul:
        return Assessment("supported", f"{len(eff)} stud{'y reports' if len(eff)==1 else 'ies report'} an effect ({desc(eff)}); none report a null result.", studies)
    if nul and not eff:
        return Assessment("not_supported", f"{len(nul)} stud{'y reports' if len(nul)==1 else 'ies report'} no effect ({desc(nul)}).", studies)
    top = studies[0]
    return Assessment(
        "conflicting",
        f"Studies disagree. Effect reported by: {desc(eff)}. No effect reported by: {desc(nul)}. "
        f"Highest-ranked design ({top.study_type}) reports '{top.stance}'. Weigh by design, sample and population.",
        studies,
    )
