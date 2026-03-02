"""Tests for experiment runner metrics parsing."""

from __future__ import annotations

import unittest

from src.cli.experiments import _parse_metrics


class ExperimentParsingTests(unittest.TestCase):
    def test_parse_metrics_extracts_expected_values(self) -> None:
        output = """
        Mean NDCG@10: 0.8123
        Mean Recall@10: 0.7567
        Median NDCG@10: 0.8012
        Median Recall@10: 0.7444
        """

        metrics = _parse_metrics(output)

        self.assertEqual(metrics["mean_ndcg_at_10"], 0.8123)
        self.assertEqual(metrics["mean_recall_at_10"], 0.7567)
        self.assertEqual(metrics["median_ndcg_at_10"], 0.8012)
        self.assertEqual(metrics["median_recall_at_10"], 0.7444)


if __name__ == "__main__":
    unittest.main()
