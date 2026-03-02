"""Utilities for generating lightweight recommendation explanations."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.core.embedding_ranking import compute_constraint_score


def _match_reasons(product_row: pd.Series, constraints: Dict[str, Any]) -> List[str]:
    reasons: list[str] = []

    if not constraints:
        return ["Ranked by overall query relevance."]

    category = constraints.get("category")
    if category and product_row.get("category") == category:
        reasons.append(f"Category match: {category}.")

    temperature = constraints.get("temperature")
    if temperature and product_row.get("temperature") == temperature:
        reasons.append(f"Temperature match: {temperature}.")

    caffeine_level = constraints.get("caffeine_level")
    if caffeine_level:
        caffeine = float(product_row.get("caffeine_mg", 0) or 0)
        if caffeine_level == "none" and caffeine == 0:
            reasons.append("Caffeine requirement met: no caffeine.")
        elif caffeine_level == "low" and 1 <= caffeine <= 50:
            reasons.append("Caffeine requirement met: low caffeine.")
        elif caffeine_level == "medium" and 51 <= caffeine <= 150:
            reasons.append("Caffeine requirement met: medium caffeine.")
        elif caffeine_level == "high" and caffeine >= 151:
            reasons.append("Caffeine requirement met: high caffeine.")

    max_price = constraints.get("max_price")
    if max_price is not None:
        price = float(product_row.get("price", 0) or 0)
        if price <= max_price:
            reasons.append(f"Within price limit: ${price:.2f} <= ${max_price:.2f}.")

    max_calories = constraints.get("max_calories")
    if max_calories is not None:
        calories = float(product_row.get("calories", 0) or 0)
        if calories <= max_calories:
            reasons.append(f"Within calorie limit: {int(calories)} <= {int(max_calories)}.")

    max_sugar = constraints.get("max_sugar")
    if max_sugar is not None:
        sugar = float(product_row.get("sugar_g", 0) or 0)
        if sugar <= max_sugar:
            reasons.append(f"Within sugar limit: {int(sugar)}g <= {int(max_sugar)}g.")

    if constraints.get("dairy_free") is True and not bool(product_row.get("contains_dairy", False)):
        reasons.append("Meets dairy-free requirement.")

    if constraints.get("vegan") is True and bool(product_row.get("is_vegan", False)):
        reasons.append("Meets vegan requirement.")

    if not reasons:
        reasons.append("Selected by semantic/lexical relevance after filtering.")

    return reasons


def explain_ranked_products(
    ranked_product_ids: List[str],
    products_df: pd.DataFrame,
    constraints: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Build explanation objects for top ranked products."""
    if top_n <= 0:
        return []

    product_map = products_df.set_index("product_id", drop=False)
    explanation_items: list[dict[str, Any]] = []

    for rank, product_id in enumerate(ranked_product_ids[:top_n], start=1):
        if product_id not in product_map.index:
            continue

        row = product_map.loc[product_id]
        reasons = _match_reasons(row, constraints)

        explanation_items.append(
            {
                "rank": rank,
                "product_id": product_id,
                "product_name": str(row.get("name", "")),
                "constraint_score": round(float(compute_constraint_score(row, constraints or {})), 4),
                "reasons": reasons,
            }
        )

    return explanation_items
