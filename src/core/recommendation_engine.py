"""Reusable recommendation engine for CLI, experiments, and API service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import EXPLANATION_TOP_N, PRODUCTS_CSV, TOP_K_PRODUCTS
from src.core.constraint_extraction import extract_constraints
from src.core.embedding_ranking import get_product_embeddings, rank_products
from src.core.explainability import explain_ranked_products
from src.core.product_filtering import filter_products


class RecommendationEngine:
    """Core service class that encapsulates retrieval and ranking logic."""

    def __init__(
        self,
        products_df: pd.DataFrame,
        product_embeddings,
        product_ids: List[str],
    ) -> None:
        self.products_df = products_df
        self.product_embeddings = product_embeddings
        self.product_ids = product_ids
        self._product_map = products_df.set_index("product_id", drop=False)

    @classmethod
    def from_config(cls) -> "RecommendationEngine":
        products_df = pd.read_csv(PRODUCTS_CSV)
        product_embeddings, product_ids = get_product_embeddings(products_df)
        return cls(products_df=products_df, product_embeddings=product_embeddings, product_ids=product_ids)

    def _to_product_item(self, product_id: str, rank: int) -> Optional[Dict[str, Any]]:
        if product_id not in self._product_map.index:
            return None

        row = self._product_map.loc[product_id]
        return {
            "rank": rank,
            "product_id": product_id,
            "name": str(row.get("name", "")),
            "category": str(row.get("category", "")),
            "temperature": str(row.get("temperature", "")),
            "price": float(row.get("price", 0.0) or 0.0),
        }

    def recommend(
        self,
        query_text: str,
        query_id: Optional[str] = None,
        top_k: int = TOP_K_PRODUCTS,
        include_explanations: bool = False,
        explanation_top_n: int = EXPLANATION_TOP_N,
    ) -> Dict[str, Any]:
        constraints = extract_constraints(query_text)
        candidates = filter_products(self.products_df, constraints)

        if candidates.empty:
            candidates = self.products_df

        ranked_product_ids = rank_products(
            query_text=query_text,
            candidate_products=candidates,
            product_embeddings=self.product_embeddings,
            product_ids=self.product_ids,
            constraints=constraints,
            top_k=top_k,
        )

        items: List[Dict[str, Any]] = []
        for rank, product_id in enumerate(ranked_product_ids, start=1):
            item = self._to_product_item(product_id, rank)
            if item is not None:
                items.append(item)

        result: Dict[str, Any] = {
            "query_id": query_id,
            "query_text": query_text,
            "constraints": constraints,
            "candidate_count": int(len(candidates)),
            "product_ids": ranked_product_ids,
            "recommendations": items,
        }

        if include_explanations:
            result["top_explanations"] = explain_ranked_products(
                ranked_product_ids=ranked_product_ids,
                products_df=self.products_df,
                constraints=constraints,
                top_n=explanation_top_n,
            )

        return result

    def recommend_batch(
        self,
        queries: List[Dict[str, str]],
        top_k: int = TOP_K_PRODUCTS,
        include_explanations: bool = False,
        explanation_top_n: int = EXPLANATION_TOP_N,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in queries:
            results.append(
                self.recommend(
                    query_text=str(item.get("query_text", "")),
                    query_id=item.get("query_id"),
                    top_k=top_k,
                    include_explanations=include_explanations,
                    explanation_top_n=explanation_top_n,
                )
            )
        return results
