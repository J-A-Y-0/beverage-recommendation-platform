"""Tests for filtering fallback and offline ranking behavior."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.embedding_ranking import rank_products
from src.core.product_filtering import filter_products


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"


class FilteringAndRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products_df = pd.read_csv(DATA_PATH)

    def test_filter_fallback_relaxes_price(self) -> None:
        constraints = {
            "category": "espresso",
            "max_price": 1.0,
            "temperature": None,
            "caffeine_level": None,
            "max_calories": None,
            "max_sugar": None,
            "dairy_free": None,
            "vegan": None,
        }

        filtered = filter_products(self.products_df, constraints)

        self.assertGreater(len(filtered), 0)
        self.assertTrue((filtered["category"] == "espresso").all())

    def test_offline_ranking_returns_top_k(self) -> None:
        constraints = {
            "category": "espresso",
            "temperature": "iced",
            "caffeine_level": None,
            "max_calories": None,
            "max_sugar": 15.0,
            "max_price": None,
            "dairy_free": None,
            "vegan": None,
        }

        candidates = filter_products(self.products_df, constraints)
        ranked = rank_products(
            query_text="iced espresso under 15g sugar",
            candidate_products=candidates,
            product_embeddings=np.empty((0, 0), dtype=float),
            product_ids=[],
            constraints=constraints,
            top_k=3,
        )

        self.assertEqual(len(ranked), min(3, len(candidates)))
        self.assertTrue(set(ranked).issubset(set(candidates["product_id"].tolist())))


if __name__ == "__main__":
    unittest.main()
