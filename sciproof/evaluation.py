"""Module 2/6: the evaluation harness. This is what separates a project from a demo.

Retrieval metrics (answerable questions):
  hit@k     any relevant chunk in the top k
  recall@k  share of relevant chunks found in the top k
  MRR       1 / rank of the first relevant chunk (how high the first good result sits)
End-to-end metrics:
  answer_rate         answerable questions that got an answer (low = over-cautious)
  abstention_accuracy unanswerable questions correctly refused (low = hallucination risk)
  citation_precision  cited passages that are in the gold sections of gold papers
  paper_precision     cited passages that come from a gold paper (right study, any section)
  faithfulness        claims that passed verification (post-hoc, on the final answer)
  verdict_accuracy    evidence verdict matches the expected one (where labelled)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from .loader import load_papers
from .pipeline import Sciproof
from .retrieval import MODES


def load_gold(path):
    return json.loads(Path(path).read_text())


def relevant_ids(chunks, item) -> set[str]:
    return {c.chunk_id for c in chunks
            if c.paper_id in item["relevant_papers"] and c.section in item["relevant_sections"]}


def retrieval_metrics(system: Sciproof, gold, k: int = 5) -> dict:
    hit, rec, rr = [], [], []
    for item in (g for g in gold if g["answerable"]):
        rel = relevant_ids(system.chunks, item)
        ranked = [h.chunk.chunk_id for h in system.retriever.search(item["question"], k)]
        found = [i for i, cid in enumerate(ranked) if cid in rel]
        hit.append(1.0 if found else 0.0)
        rec.append(len({ranked[i] for i in found}) / len(rel) if rel else 0.0)
        rr.append(1.0 / (found[0] + 1) if found else 0.0)
    return {f"hit@{k}": mean(hit), f"recall@{k}": mean(rec), "MRR": mean(rr)}


def end_to_end_metrics(system: Sciproof, gold) -> dict:
    ans_rate, abst, cprec, pprec, faith, verd = [], [], [], [], [], []
    for item in gold:
        a = system.ask(item["question"])
        if not item["answerable"]:
            abst.append(1.0 if a.abstained else 0.0)
            continue
        ans_rate.append(0.0 if a.abstained else 1.0)
        if a.abstained:
            continue
        rel = relevant_ids(system.chunks, item)
        cited = [cid for c in a.claims for cid in c.chunk_ids]
        cprec.append(sum(cid in rel for cid in cited) / len(cited))
        pprec.append(sum(system.by_id[cid].paper_id in item["relevant_papers"] for cid in cited) / len(cited))
        faith.append(mean(1.0 if c.verified else 0.0 for c in a.claims))
        if "expected_verdict" in item and a.assessment:
            verd.append(1.0 if a.assessment.verdict == item["expected_verdict"] else 0.0)
    f = lambda xs: mean(xs) if xs else float("nan")
    return {"answer_rate": f(ans_rate), "abstention_accuracy": f(abst),
            "citation_precision": f(cprec), "paper_precision": f(pprec), "faithfulness": f(faith), "verdict_accuracy": f(verd)}


def run(papers, gold, **kw) -> dict:
    s = Sciproof(papers, **kw)
    return {**retrieval_metrics(s, gold), **end_to_end_metrics(s, gold)}


def ablation_report(corpus="data/synthetic/papers.json", gold_path="data/eval/gold.json") -> str:
    papers, gold = load_papers(corpus), load_gold(gold_path)
    cols = ["hit@5", "recall@5", "MRR", "answer_rate", "abstention_accuracy", "citation_precision", "paper_precision", "faithfulness", "verdict_accuracy"]

    def table(title, rows):
        out = [f"### {title}", "", "| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name, m in rows:
            out.append(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
        return "\n".join(out) + "\n"

    parts = [table("A. Retrieval strategy (600-char chunks, title context on)",
                   [(m, run(papers, gold, mode=m)) for m in MODES])]
    parts.append(table("B. Chunk size (mode = bm25)",
                       [(f"{n} chars", run(papers, gold, mode="bm25", max_chars=n)) for n in (150, 300, 600, 1000)]))
    parts.append(table("C. Contextual indexing: prepend paper title? (mode = bm25)",
                       [("title context OFF", run(papers, gold, mode="bm25", use_title=False)),
                        ("title context ON", run(papers, gold, mode="bm25", use_title=True))]))
    parts.append(table("D. Same, with the full hybrid_rerank pipeline",
                       [("title context OFF", run(papers, gold, mode="hybrid_rerank", use_title=False)),
                        ("title context ON", run(papers, gold, mode="hybrid_rerank", use_title=True))]))
    return "\n".join(parts)


if __name__ == "__main__":
    print(ablation_report())
