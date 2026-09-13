from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from translation_gold import REQUIRED_FEATURES, load_gold_cases, validate_gold_cases  # noqa: E402


class TranslationGoldTests(unittest.TestCase):
    def test_public_gold_corpus_contract(self) -> None:
        cases = load_gold_cases(TESTS / "fixtures" / "translation_gold_v1.json")
        errors = validate_gold_cases(cases)
        self.assertEqual([], errors)
        self.assertGreaterEqual(len(cases), 120)
        covered = {feature for case in cases for feature in case["features"]}
        self.assertTrue(REQUIRED_FEATURES <= covered)

    def test_ids_are_unique_and_references_are_original_synthetic_content(self) -> None:
        cases = load_gold_cases(TESTS / "fixtures" / "translation_gold_v1.json")
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for case in cases:
            self.assertTrue(case["source"].strip())
            self.assertTrue(case["reference_vi"].strip())
            self.assertIn("structural_gold", case)
            self.assertIn("rubric", case)


if __name__ == "__main__":
    unittest.main()
