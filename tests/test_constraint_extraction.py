"""Tests for rule-based constraint extraction."""

from __future__ import annotations

import unittest

from src.core.constraint_extraction import _rule_based_extract


class ConstraintExtractionTests(unittest.TestCase):
    def test_rule_based_extract_multiple_constraints(self) -> None:
        query = "need an iced vegan latte under $5 and 15g sugar or less"
        constraints = _rule_based_extract(query)

        self.assertEqual(constraints["category"], "espresso")
        self.assertEqual(constraints["temperature"], "iced")
        self.assertEqual(constraints["vegan"], True)
        self.assertAlmostEqual(constraints["max_price"], 5.0)
        self.assertAlmostEqual(constraints["max_sugar"], 15.0)

    def test_rule_based_extract_caffeine_none(self) -> None:
        query = "i want caffeine free herbal tea without milk"
        constraints = _rule_based_extract(query)

        self.assertEqual(constraints["category"], "tea")
        self.assertEqual(constraints["caffeine_level"], "none")
        self.assertEqual(constraints["dairy_free"], True)


if __name__ == "__main__":
    unittest.main()
