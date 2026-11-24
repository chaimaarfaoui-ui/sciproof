import unittest
from pathlib import Path

from sciproof.chunking import chunk_paper, chunk_section, split_sentences
from sciproof.loader import load_papers

DATA = Path(__file__).parent.parent / "data" / "sample" / "papers.json"


class ChunkingTests(unittest.TestCase):
    def test_abbreviations_do_not_split_sentences(self):
        text = "Randomization is shown in Fig. 1. Data were analysed e.g. by ITT. Done."
        self.assertEqual(
            split_sentences(text),
            [
                "Randomization is shown in Fig. 1.",
                "Data were analysed e.g. by ITT.",
                "Done.",
            ],
        )

    def test_chunks_respect_max_chars_and_never_split_sentences(self):
        sentences = [f"Sentence number {i} is here." for i in range(20)]
        chunks = chunk_section(sentences, max_chars=100, overlap_sentences=0)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), " ".join(sentences).split())

    def test_overlap_repeats_boundary_sentence(self):
        sentences = ["Alpha is first.", "Beta is second.", "Gamma is third.", "Delta is fourth."]
        chunks = chunk_section(sentences, max_chars=35, overlap_sentences=1)
        self.assertGreaterEqual(len(chunks), 2)
        # the last sentence of chunk 0 must reappear at the start of chunk 1
        last_of_first = chunks[0].split(". ")[-1].rstrip(".")
        self.assertIn(last_of_first, chunks[1])

    def test_oversized_sentence_becomes_its_own_chunk(self):
        long = "X" * 500 + "."
        self.assertEqual(chunk_section([long], max_chars=100), [long])

    def test_no_duplicate_trailing_chunk(self):
        # exact-fit input must not produce an overlap-only trailing chunk
        sentences = ["Aaaa bbbb.", "Cccc dddd."]
        chunks = chunk_section(sentences, max_chars=21, overlap_sentences=1)
        self.assertEqual(chunks, ["Aaaa bbbb. Cccc dddd."])

    def test_references_are_skipped_and_ids_are_stable(self):
        papers = load_papers(DATA)
        chunks = chunk_paper(papers[0], max_chars=300)
        self.assertTrue(all(c.section != "References" for c in chunks))
        self.assertEqual(chunks[0].chunk_id, "SYN-001::abstract::0")
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_metadata_is_attached(self):
        papers = load_papers(DATA)
        chunks = chunk_paper(papers[1])
        self.assertEqual({c.study_type for c in chunks}, {"cohort"})
        self.assertEqual({c.year for c in chunks}, {2023})


if __name__ == "__main__":
    unittest.main()
