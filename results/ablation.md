### A. Retrieval strategy (600-char chunks, title context on)

| config | hit@5 | recall@5 | MRR | answer_rate | abstention_accuracy | citation_precision | paper_precision | faithfulness | verdict_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 1.00 | 0.88 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | 1.00 | 0.83 | 0.55 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid | 1.00 | 0.86 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid_rerank | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### B. Chunk size (mode = bm25)

| config | hit@5 | recall@5 | MRR | answer_rate | abstention_accuracy | citation_precision | paper_precision | faithfulness | verdict_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 150 chars | 1.00 | 0.43 | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 300 chars | 1.00 | 0.86 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 600 chars | 1.00 | 0.88 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1000 chars | 1.00 | 0.88 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### C. Contextual indexing: prepend paper title? (mode = bm25)

| config | hit@5 | recall@5 | MRR | answer_rate | abstention_accuracy | citation_precision | paper_precision | faithfulness | verdict_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| title context OFF | 1.00 | 0.59 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| title context ON | 1.00 | 0.88 | 0.89 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### D. Same, with the full hybrid_rerank pipeline

| config | hit@5 | recall@5 | MRR | answer_rate | abstention_accuracy | citation_precision | paper_precision | faithfulness | verdict_accuracy |
|---|---|---|---|---|---|---|---|---|---|
| title context OFF | 1.00 | 0.61 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| title context ON | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
