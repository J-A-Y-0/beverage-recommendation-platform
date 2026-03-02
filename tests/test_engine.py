"""Tests for recommendation engine service layer."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.recommendation_engine import RecommendationEngine


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"


class RecommendationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        products_df = pd.read_csv(DATA_PATH)
        cls.engine = RecommendationEngine(
            products_df=products_df,
            product_embeddings=np.empty((0, 0), dtype=float),
            product_ids=[],
        )

    def test_recommend_returns_structured_payload(self) -> None:
        result = self.engine.recommend(
            query_text="need an iced espresso under 15g sugar",
            include_explanations=True,
            explanation_top_n=2,
            top_k=3,
        )

        self.assertIn("constraints", result)
        self.assertIn("candidate_count", result)
        self.assertIn("product_ids", result)
        self.assertIn("recommendations", result)
        self.assertIn("top_explanations", result)
        self.assertGreater(len(result["product_ids"]), 0)

    def test_recommend_batch_works(self) -> None:
        batch = [
            {"query_id": "Q1", "query_text": "hot brewed coffee"},
            {"query_id": "Q2", "query_text": "vegan iced espresso"},
        ]
        results = self.engine.recommend_batch(batch, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["query_id"], "Q1")
        self.assertEqual(results[1]["query_id"], "Q2")


if __name__ == "__main__":
    unittest.main()
