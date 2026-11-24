"""Run the Module 1 pipeline end to end and print what it produced."""
from sciproof.chunking import chunk_paper
from sciproof.loader import load_papers

papers = load_papers("data/sample/papers.json")
all_chunks = [c for p in papers for c in chunk_paper(p, max_chars=1000)]
print(f"{len(papers)} papers -> {len(all_chunks)} chunks\n")
for c in all_chunks:
    print(f"[{c.chunk_id}] ({c.study_type}, {c.year}) {len(c.text)} chars")
    print(f"   {c.text}\n")
