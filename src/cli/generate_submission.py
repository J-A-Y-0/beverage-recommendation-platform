"""Generate submission CSV for test queries."""

from __future__ import annotations

import json
import os

import pandas as pd

from src.config import (
    EXPLANATIONS_JSON,
    EXPLANATION_TOP_N,
    EMBEDDING_MODEL,
    EXPANSION_MODEL,
    EXTRACTION_MODEL,
    HAS_OPENAI_API,
    MODEL_PRESET,
    QUERIES_TEST_CSV,
    SUBMISSION_CSV,
    TOP_K_PRODUCTS,
    USE_QUERY_EXPANSION,
    WRITE_EXPLANATIONS,
)
from src.core.recommendation_engine import RecommendationEngine


def main() -> None:
    """Generate submission file for test queries."""
    print("Loading data...")
    queries_df = pd.read_csv(QUERIES_TEST_CSV)
    engine = RecommendationEngine.from_config()

    print(f"Loaded {len(engine.products_df)} products and {len(queries_df)} test queries")
    mode = "online (OpenAI)" if HAS_OPENAI_API else "offline (rule-based + lexical)"
    print(f"Execution mode: {mode}")
    print(f"Model preset: {MODEL_PRESET}")
    print(f"Extraction model: {EXTRACTION_MODEL}")
    print(f"Expansion model: {EXPANSION_MODEL} (enabled={USE_QUERY_EXPANSION})")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Write explanations: {WRITE_EXPLANATIONS}")

    print("\nProcessing test queries...")
    results: list[dict[str, str]] = []
    explanations_payload: list[dict[str, object]] = []

    fallback_products = engine.products_df["product_id"].tolist()[:TOP_K_PRODUCTS]
    fallback_products_str = ";".join(fallback_products)

    for idx, row in queries_df.iterrows():
        query_id = str(row["query_id"])
        query_text = str(row["query_text"])

        print(f"\n[{idx + 1}/{len(queries_df)}] {query_id}: {query_text[:60]}...")

        try:
            result = engine.recommend(
                query_id=query_id,
                query_text=query_text,
                top_k=TOP_K_PRODUCTS,
                include_explanations=WRITE_EXPLANATIONS,
                explanation_top_n=EXPLANATION_TOP_N,
            )
            ranked_products = result["product_ids"]

            print(f"  Constraints: {result['constraints']}")
            print(f"  Candidates: {result['candidate_count']} products")
            print(f"  Top 5: {ranked_products[:5]}")

            results.append({"query_id": query_id, "products": ";".join(ranked_products)})

            if WRITE_EXPLANATIONS:
                explanations_payload.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "constraints": result.get("constraints", {}),
                        "top_explanations": result.get("top_explanations", []),
                    }
                )

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"query_id": query_id, "products": fallback_products_str})
            if WRITE_EXPLANATIONS:
                explanations_payload.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "constraints": {},
                        "top_explanations": [],
                        "error": str(exc),
                    }
                )

    submission_df = pd.DataFrame(results)

    if len(submission_df) != len(queries_df):
        print(f"\nWARNING: Expected {len(queries_df)} queries, got {len(submission_df)}")
        existing_ids = set(submission_df["query_id"])
        for _, row in queries_df.iterrows():
            query_id = str(row["query_id"])
            if query_id not in existing_ids:
                submission_df = pd.concat(
                    [submission_df, pd.DataFrame([{"query_id": query_id, "products": fallback_products_str}])],
                    ignore_index=True,
                )

    submission_df = submission_df.sort_values("query_id").reset_index(drop=True)

    output_dir = os.path.dirname(SUBMISSION_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    submission_df.to_csv(SUBMISSION_CSV, index=False)

    if WRITE_EXPLANATIONS:
        explanation_dir = os.path.dirname(EXPLANATIONS_JSON)
        if explanation_dir:
            os.makedirs(explanation_dir, exist_ok=True)
        with open(EXPLANATIONS_JSON, "w", encoding="utf-8") as file_obj:
            json.dump(explanations_payload, file_obj, indent=2)
        print(f"Explanations saved to {EXPLANATIONS_JSON}")

    print(f"\n{'=' * 60}")
    print(f"Submission saved to {SUBMISSION_CSV}")
    print(f"Total queries: {len(submission_df)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
