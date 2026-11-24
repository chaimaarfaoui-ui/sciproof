import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from sciproof.app import handle_ask, handle_check, make_handler
from sciproof.evaluation import retrieval_metrics
from sciproof.ingest import parse_plain_text, parse_pubmed_xml
from sciproof.models import Chunk
from sciproof.pipeline import Sciproof

CORPUS = Path(__file__).parent.parent / "data" / "synthetic" / "papers.json"

XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>A toy title</ArticleTitle><Abstract>
<AbstractText Label="BACKGROUND">Background text.</AbstractText>
<AbstractText Label="RESULTS">Result text with p = 0.01.</AbstractText></Abstract>
<PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


class IngestTests(unittest.TestCase):
    def test_pubmed_xml_parsed_and_abstractless_skipped(self):
        papers = parse_pubmed_xml(XML)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual((p.paper_id, p.year, p.study_type), ("PMID111", 2020, "RCT"))
        self.assertEqual([s.name for s in p.sections], ["Background", "Results"])

    def test_plain_text_split_on_headings(self):
        text = "Abstract\nShort.\n\n1. Methods\nWe did things.\n\nResults\nIt worked.\n\nReferences\n1. x"
        p = parse_plain_text(text, "T1", "t", 2020)
        self.assertEqual([s.name for s in p.sections], ["Abstract", "Methods", "Results", "References"])

    def test_plain_text_without_headings_is_one_section(self):
        self.assertEqual(parse_plain_text("just text", "T", "t", 1).sections[0].name, "Full Text")


class MetricTests(unittest.TestCase):
    def test_recall_and_mrr_are_computed_correctly(self):
        mk = lambda i: Chunk(f"c{i}", "P", "Results", i, "t", 2020, "RCT")
        chunks = [mk(i) for i in range(4)]
        ranked = [SimpleNamespace(chunk=chunks[i]) for i in (2, 0, 1)]   # relevant = c0,c1 -> first at rank 2
        sys_ = SimpleNamespace(chunks=chunks, retriever=SimpleNamespace(search=lambda q, k: ranked[:k]))
        gold = [{"question": "q", "answerable": True, "relevant_papers": ["P"], "relevant_sections": ["Results"]}]
        # every chunk is relevant here, so narrow it: make c2,c3 a different paper
        chunks[2] = Chunk("c2", "Q", "Results", 2, "t", 2020, "RCT")
        chunks[3] = Chunk("c3", "Q", "Results", 3, "t", 2020, "RCT")
        ranked[0] = SimpleNamespace(chunk=chunks[2])
        m = retrieval_metrics(sys_, gold, k=3)
        self.assertAlmostEqual(m["MRR"], 0.5)
        self.assertAlmostEqual(m["recall@3"], 1.0)
        self.assertEqual(m["hit@3"], 1.0)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = Sciproof.from_file(CORPUS)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.system))
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def url(self, p):
        return f"http://127.0.0.1:{self.port}{p}"

    def post(self, path, payload):
        req = urllib.request.Request(self.url(path), json.dumps(payload).encode(), {"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    def test_index_page_and_papers(self):
        self.assertIn(b"Sciproof", urllib.request.urlopen(self.url("/")).read())
        self.assertEqual(len(json.loads(urllib.request.urlopen(self.url("/api/papers")).read())["papers"]), 19)

    def test_ask_endpoint_returns_cited_answer(self):
        d = self.post("/api/ask", {"question": "Does ZX-101 reduce resting heart rate?"})
        self.assertFalse(d["abstained"])
        self.assertTrue(all(cid in d["passages"] for c in d["claims"] for cid in c["chunk_ids"]))

    def test_ask_endpoint_abstains(self):
        self.assertTrue(self.post("/api/ask", {"question": "How does photosynthesis work in algae?"})["abstained"])

    def test_passage_endpoint(self):
        cid = self.post("/api/ask", {"question": "Does ZX-101 reduce resting heart rate?"})["claims"][0]["chunk_ids"][0]
        d = json.loads(urllib.request.urlopen(self.url("/api/passage?id=" + cid)).read())
        self.assertEqual(d["chunk_id"], cid)

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            handle_ask(self.system, {"question": ""})
        with self.assertRaises(ValueError):
            handle_check(self.system, {"claim": "x" * 5000})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/api/ask", {"question": ""})
        self.assertEqual(cm.exception.code, 400)

    def test_unknown_route_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.url("/nope"))
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
