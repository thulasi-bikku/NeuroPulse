"""Tests for the reference checker (offline, with mocked Crossref records)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_references as C  # noqa: E402


class CheckerTests(unittest.TestCase):
    def test_bibliography_parses_and_dois_are_well_formed(self):
        entries = C.parse_bib((ROOT / "paper" / "references.bib").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(entries), 60)
        keys = [e["key"] for e in entries]
        self.assertEqual(len(keys), len(set(keys)))
        for e in entries:
            if e["doi"]:
                self.assertRegex(e["doi"], r"^10\.\d{4,9}/\S+$", e["key"])

    def test_retraction_is_detected(self):
        self.assertEqual(C.retraction_status({"title": ["A study"], "updated-by": [{"type": "retraction", "DOI": "10.1/x"}]}),
                         ["retraction notice 10.1/x"])
        self.assertTrue(C.retraction_status({"title": ["Retracted: A study"]}))
        self.assertTrue(C.retraction_status({"title": ["Notice"], "update-to": [{"type": "withdrawal", "DOI": "10.1/y"}]}))

    def test_clean_record_and_ordinary_corrections_pass(self):
        self.assertEqual(C.retraction_status({"title": ["A study"]}), [])
        self.assertEqual(C.retraction_status({"title": ["A study"], "updated-by": [{"type": "correction", "DOI": "10.1/z"}]}), [])

    def test_title_matching(self):
        self.assertTrue(C.title_matches("Deep learning-based {EEG} emotion recognition",
                                        "Deep learning-based EEG emotion recognition: Current trends"))
        self.assertFalse(C.title_matches("Attention is all you need", "Completely different subject matter here"))


if __name__ == "__main__":
    unittest.main()
