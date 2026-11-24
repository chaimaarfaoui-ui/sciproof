import unittest
from pathlib import Path

from sciproof.chunking import chunk_paper
from sciproof.loader import load_papers
from sciproof.retrieval import MODES, Retriever
from sciproof.retrieval.bm25 import BM25Index
from sciproof.retrieval.dense import DenseIndex, LSAEmbedder
from sciproof.text import content_terms, entity_terms, stem, tokenize

CORPUS = Path(__file__).parent.parent / "data" / "synthetic" / "papers.json"


class TextTests(unittest.TestCase):
    def test_stemming_unifies_verb_forms(self):
        self.assertEqual(stem("lowers"), stem("lowering"))
        self.assertEqual(stem("reduce"), stem("reduced"))

    def test_entities_are_detected(self):
        self.assertEqual(entity_terms("effect of ZX-101 on IL-6"), {"zx", "101", "il", "6"})
        self.assertEqual(entity_terms("does this reduce heart rate"), set())

    def test_question_words_dropped_from_content_terms(self):
        self.assertNotIn("effect", content_terms("What was the effect of X"))
        self.assertIn("effect", tokenize("What was the effect of X"))


class BM25Tests(unittest.TestCase):
    def test_rare_term_outweighs_common_term(self):
        idx = BM25Index(["heart rate study", "heart rate trial", "heart rate zebrafish"])
        self.assertGreater(idx.idf("zebrafish"), idx.idf("heart"))
        self.assertEqual(idx.search("zebrafish heart", 1)[0][0], 2)

    def test_unknown_term_scores_nothing(self):
        self.assertEqual(BM25Index(["a b c"]).search("zzzz"), [])


class DenseTests(unittest.TestCase):
    DOCS = ["pulse and heartbeat measured", "heartbeat and heart rate measured",
            "glucose and insulin measured", "insulin and blood sugar measured"]

    def test_compression_bridges_synonyms_that_bm25_cannot(self):
        """Doc 0 shares NO words with the query 'heart rate'. BM25 can't find it;
        a compressed (dim=2) LSA space can, via co-occurrence with 'heartbeat'."""
        bm25_ids = {i for i, _ in BM25Index(self.DOCS).search("heart rate", 4)}
        dense_ids = [i for i, _ in DenseIndex(self.DOCS, LSAEmbedder(dim=2)).search("heart rate", 2)]
        self.assertNotIn(0, bm25_ids)
        self.assertIn(0, dense_ids)

    def test_without_compression_dense_is_no_better_than_keywords(self):
        """dim >= rank keeps every dimension, so no generalisation happens."""
        top = {i: s for i, s in DenseIndex(self.DOCS, LSAEmbedder(dim=48)).search("heart rate", 4)}
        self.assertLess(top.get(0, 0.0), 1e-6)


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = [c for p in load_papers(CORPUS) for c in chunk_paper(p)]

    def test_all_modes_return_k_hits(self):
        for m in MODES:
            hits = Retriever(self.chunks, m).search("systolic blood pressure", 5)
            self.assertEqual(len(hits), 5, m)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            Retriever(self.chunks, "magic")

    def test_title_context_lets_results_chunk_match_compound(self):
        """Results text never names the compound; only the title does."""
        q = "VL-9 systolic blood pressure"
        on = Retriever(self.chunks, "bm25", use_title=True).search(q, 5)
        off = Retriever(self.chunks, "bm25", use_title=False).search(q, 5)
        res_on = [h for h in on if h.chunk.section == "Results" and "VL-9" in h.chunk.title]
        res_off = [h for h in off if h.chunk.section == "Results" and "VL-9" in h.chunk.title]
        self.assertGreater(len(res_on), len(res_off))

    def test_coverage_flags_missing_entity(self):
        r = Retriever(self.chunks)
        ok, _ = r.coverage("effect of VL-9 on systolic blood pressure", r.search("VL-9 systolic blood pressure", 10))
        bad, missing = r.coverage("effect of ZX-999 on resting heart rate", r.search("ZX-999 resting heart rate", 10))
        self.assertGreaterEqual(ok, 0.8)
        self.assertLess(bad, 0.8)
        self.assertIn("999", missing)


if __name__ == "__main__":
    unittest.main()
