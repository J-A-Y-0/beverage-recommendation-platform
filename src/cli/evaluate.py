"""Evaluate the recommendation pipeline on training queries."""

from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_MODEL,
    EXPANSION_MODEL,
    EXTRACTION_MODEL,
    HAS_OPENAI_API,
    MODEL_PRESET,
    QUERIES_TRAIN_CSV,
    USE_QUERY_EXPANSION,
)
from src.core.recommendation_engine import RecommendationEngine


def ndcg_at_k(relevant: List[str], predicted: List[str], k: int = 10) -> float:
    """Compute NDCG@k."""
    if not relevant:
        return 0.0

    pred_k = predicted[:k]

    dcg = 0.0
    for i, pid in enumerate(pred_k):
        if pid in relevant:
            dcg += 1.0 / np.log2(i + 2)

    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_k(relevant: List[str], predicted: List[str], k: int = 10) -> float:
    """Compute Recall@k."""
    if not relevant:
        return 0.0

    pred_k = predicted[:k]
    return len(set(relevant) & set(pred_k)) / len(set(relevant))


def _parse_relevant_products(raw: object) -> List[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []

    if isinstance(raw, list):
        return [str(item) for item in raw]

    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            return []

    return []


def main() -> None:
    """Run evaluation on training queries."""
    print("Loading data...")
    queries_df = pd.read_csv(QUERIES_TRAIN_CSV)
    engine = RecommendationEngine.from_config()

    print(f"Loaded {len(engine.products_df)} products and {len(queries_df)} training queries")
    mode = "online (OpenAI)" if HAS_OPENAI_API else "offline (rule-based + lexical)"
    print(f"Execution mode: {mode}")
    print(f"Model preset: {MODEL_PRESET}")
    print(f"Extraction model: {EXTRACTION_MODEL}")
    print(f"Expansion model: {EXPANSION_MODEL} (enabled={USE_QUERY_EXPANSION})")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    print("\nEvaluating on training queries...")
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []

    for idx, row in queries_df.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["query_text"])
        relevant_products = _parse_relevant_products(row.get("relevant_products"))

        print(f"\n[{idx + 1}/{len(queries_df)}] {query_id}: {query_text[:60]}...")

        try:
            result = engine.recommend(query_text=query_text)
            ranked_products = result["product_ids"]

            print(f"  Constraints: {result['constraints']}")
            print(f"  Candidates: {result['candidate_count']} products")

            ndcg = ndcg_at_k(relevant_products, ranked_products, k=10)
            recall = recall_at_k(relevant_products, ranked_products, k=10)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)

            print(f"  NDCG@10: {ndcg:.4f}, Recall@10: {recall:.4f}")
            print(f"  Top 5: {ranked_products[:5]}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            ndcg_scores.append(0.0)
            recall_scores.append(0.0)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean NDCG@10: {np.mean(ndcg_scores):.4f}")
    print(f"Mean Recall@10: {np.mean(recall_scores):.4f}")
    print(f"Median NDCG@10: {np.median(ndcg_scores):.4f}")
    print(f"Median Recall@10: {np.median(recall_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
