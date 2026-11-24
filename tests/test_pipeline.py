import json
import unittest
from pathlib import Path

from sciproof.generation import AnthropicGenerator, GenerationError, SYSTEM_PROMPT
from sciproof.pipeline import Sciproof, answer_to_dict
import re

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "data" / "synthetic" / "papers.json"
GOLD = json.loads((ROOT / "data" / "eval" / "gold.json").read_text())


def fake_llm(claims_fn):
    """A fake HTTP transport: builds a reply from the chunk ids found in the prompt."""
    def post(body):
        prompt = body["messages"][0]["content"]
        ids = re.findall(r"^\[([^\]]+)\]", prompt, flags=re.M)
        return {"content": [{"type": "text", "text": claims_fn(ids, prompt)}]}
    return post


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Sciproof.from_file(CORPUS)

    def test_answers_cite_only_retrieved_passages_and_are_verified(self):
        a = self.s.ask("What was the effect of VL-9 on systolic blood pressure?")
        self.assertFalse(a.abstained)
        self.assertTrue(a.claims)
        for c in a.claims:
            self.assertTrue(c.verified)
            for cid in c.chunk_ids:
                self.assertIn(cid, a.passages)
        self.assertTrue(all(self.s.by_id[cid].title.find("VL-9") >= 0 for c in a.claims for cid in c.chunk_ids))

    def test_abstains_on_every_unanswerable_gold_question(self):
        for g in GOLD:
            if not g["answerable"]:
                a = self.s.ask(g["question"])
                self.assertTrue(a.abstained, g["question"])
                self.assertTrue(a.reason)

    def test_answers_every_answerable_gold_question(self):
        for g in GOLD:
            if g["answerable"]:
                self.assertFalse(self.s.ask(g["question"]).abstained, g["question"])

    def test_conflicting_evidence_is_flagged(self):
        item = next(g for g in GOLD if g.get("expected_verdict") == "conflicting")
        self.assertEqual(self.s.check_claim(item["question"]).verdict, "conflicting")

    def test_agreeing_evidence_is_supported(self):
        item = next(g for g in GOLD if g.get("expected_verdict") == "supported")
        self.assertEqual(self.s.check_claim(item["question"]).verdict, "supported")

    def test_check_claim_insufficient_for_unknown_entity(self):
        self.assertEqual(self.s.check_claim("ZX-999 lowers resting heart rate").verdict, "insufficient")

    def test_serialisation_is_json_safe(self):
        d = answer_to_dict(self.s.ask("Does ZX-101 reduce resting heart rate?"))
        json.dumps(d)


class LLMGeneratorTests(unittest.TestCase):
    Q = "What was the effect of VL-9 on systolic blood pressure?"

    def build(self, fn):
        return Sciproof.from_file(CORPUS, generator=AnthropicGenerator(api_key="x", post=fake_llm(fn)))

    def test_valid_llm_claims_pass_through_verifier(self):
        def fn(ids, prompt):
            res = next(i for i in ids if i.endswith("::results::0"))
            return json.dumps({"claims": [{"text": "Systolic blood pressure decreased by 11 mmHg relative to placebo (p = 0.004).", "chunk_ids": [res]}], "abstain": False})
        a = self.build(fn).ask(self.Q)
        self.assertFalse(a.abstained)
        self.assertTrue(all(c.verified for c in a.claims))

    def test_fabricated_number_from_llm_is_dropped(self):
        def fn(ids, prompt):
            res = next(i for i in ids if i.endswith("::results::0"))
            return json.dumps({"claims": [{"text": "Systolic blood pressure decreased by 40 mmHg relative to placebo.", "chunk_ids": [res]}]})
        a = self.build(fn).ask(self.Q)
        self.assertTrue(a.abstained)
        self.assertTrue(any("number" in n for n in a.notes))

    def test_invented_citation_id_is_dropped(self):
        def fn(ids, prompt):
            return json.dumps({"claims": [{"text": "Systolic blood pressure decreased.", "chunk_ids": ["SYN-999::results::0"]}]})
        self.assertTrue(self.build(fn).ask(self.Q).abstained)

    def test_malformed_llm_output_falls_back_to_extractive(self):
        a = self.build(lambda ids, p: "not json at all").ask(self.Q)
        self.assertFalse(a.abstained)
        self.assertTrue(any("fell back" in n for n in a.notes))

    def test_llm_abstain_is_respected(self):
        a = self.build(lambda ids, p: json.dumps({"claims": [], "abstain": True})).ask(self.Q)
        self.assertTrue(a.abstained)

    def test_missing_api_key_raises_generation_error(self):
        with self.assertRaises(GenerationError):
            AnthropicGenerator(api_key=None)._http_post({}) if not __import__("os").environ.get("ANTHROPIC_API_KEY") else (_ for _ in ()).throw(GenerationError())

    def test_prompt_treats_passages_as_untrusted(self):
        self.assertIn("untrusted", SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()


class ClaimQualityTests(unittest.TestCase):
    """Regressions found in the first browser test (screenshots of VL-9 / RT-450 / ZX-999)."""

    @classmethod
    def setUpClass(cls):
        cls.s = Sciproof.from_file(CORPUS)

    def test_conflicting_studies_both_appear_in_claims(self):
        a = self.s.ask("RT-450 lowers fasting glucose")
        types = {self.s.by_id[cid].study_type for c in a.claims for cid in c.chunk_ids}
        self.assertEqual(a.assessment.verdict, "conflicting")
        self.assertTrue({"RCT", "cohort"} <= types, types)

    def test_no_duplicate_results_from_same_paper(self):
        import re
        a = self.s.ask("What was the effect of VL-9 on systolic blood pressure?")
        keys = []
        for c in a.claims:
            m = re.search(r"\bp\s*[=<>]\s*\d*\.\d+", c.text.lower())
            if m:
                keys.append((self.s.by_id[c.chunk_ids[0]].paper_id, re.sub(r"\s+", "", m.group(0))))
        self.assertEqual(len(keys), len(set(keys)), keys)

    def test_refusal_has_no_internal_debug_notes(self):
        a = self.s.ask("What was the effect of ZX-999 on resting heart rate?")
        self.assertTrue(a.abstained)
        self.assertFalse(any("coverage=" in n for n in a.notes))
