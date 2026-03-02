"""Tests for recommendation explanation generation."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.core.explainability import explain_ranked_products


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products_df = pd.read_csv(DATA_PATH)

    def test_explanation_contains_reasons(self) -> None:
        constraints = {
            "category": "espresso",
            "temperature": "iced",
            "caffeine_level": None,
            "max_calories": None,
            "max_sugar": 15.0,
            "max_price": 6.0,
            "dairy_free": None,
            "vegan": None,
        }
        ranked_ids = ["PUB_ICE_002", "PUB_ICE_001"]

        explanations = explain_ranked_products(ranked_ids, self.products_df, constraints, top_n=2)

        self.assertEqual(len(explanations), 2)
        self.assertEqual(explanations[0]["rank"], 1)
        self.assertEqual(explanations[0]["product_id"], "PUB_ICE_002")
        self.assertGreater(len(explanations[0]["reasons"]), 0)


if __name__ == "__main__":
    unittest.main()
