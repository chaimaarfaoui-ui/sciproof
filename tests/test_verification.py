import unittest

from sciproof.models import Chunk, Claim
from sciproof.verification import LexicalVerifier

SRC = Chunk("P::results::0", "P", "Results", 0,
            "Resting heart rate fell by 6 beats per minute in the treatment arm (p = 0.01). "
            "No serious adverse events were recorded.", 2021, "RCT", "T")
PASSAGES = {SRC.chunk_id: SRC}
V = LexicalVerifier()


class VerifierTests(unittest.TestCase):
    def check(self, text, ids=("P::results::0",)):
        return V.verify(Claim(text, tuple(ids)), PASSAGES)

    def test_faithful_claim_passes(self):
        self.assertTrue(self.check("Resting heart rate fell by 6 beats per minute (p = 0.01).").verified)

    def test_fabricated_number_fails(self):
        r = self.check("Resting heart rate fell by 9 beats per minute.")
        self.assertFalse(r.verified)
        self.assertIn("number", r.note)

    def test_unknown_citation_fails(self):
        r = self.check("Resting heart rate fell by 6 beats per minute.", ids=("P::results::7",))
        self.assertFalse(r.verified)
        self.assertIn("unknown", r.note)

    def test_missing_citation_fails(self):
        self.assertFalse(self.check("Resting heart rate fell.", ids=()).verified)

    def test_unsupported_wording_fails(self):
        r = self.check("The compound cured insomnia and improved cardiovascular mortality.")
        self.assertFalse(r.verified)

    def test_flipped_negation_fails(self):
        r = self.check("Serious adverse events were recorded.")
        self.assertFalse(r.verified)
        self.assertIn("negation", r.note)


if __name__ == "__main__":
    unittest.main()
