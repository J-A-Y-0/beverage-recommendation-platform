"""Ranking using embeddings, query expansion, and hybrid scoring."""

from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CONSTRAINT_WEIGHT,
    EMBEDDING_MODEL,
    EMBEDDING_WEIGHT,
    PRODUCT_EMBEDDINGS_CACHE,
    TOP_K_PRODUCTS,
)
from src.openai_client import get_openai_client
from src.core.query_expansion import expand_query

client = get_openai_client()
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_product_text(row: pd.Series) -> str:
    """Build text representation of a product for matching."""
    parts: list[str] = []
    name = str(row["name"])
    parts.append(name)

    if "Caffè" in name or "Caffe" in name:
        parts.append(name.replace("Caffè", "Coffee").replace("Caffe", "Coffee"))

    if pd.notna(row.get("description")):
        parts.append(str(row["description"]))

    category_info: list[str] = []
    category = row.get("category")
    if category:
        cat = str(category)
        category_info.append(cat)
        if cat == "brewed":
            category_info.extend(["drip coffee", "house coffee"])
        elif cat == "cold_brew":
            category_info.extend(["cold brew coffee", "nitro cold brew"])
        elif cat == "espresso":
            category_info.extend(["espresso drink", "espresso-based"])
        elif cat == "frappuccino":
            category_info.extend(["frozen coffee", "blended coffee"])
        elif cat == "refresher":
            category_info.extend(["refreshing drink", "fruity drink"])
        elif cat == "tea":
            category_info.extend(["tea drink", "tea beverage"])

    subcategory = row.get("subcategory")
    if subcategory:
        category_info.append(str(subcategory))

    if category_info:
        parts.append(" ".join(category_info))

    temperature = row.get("temperature")
    if temperature:
        temp = str(temperature)
        parts.append(f"{temp} drink")
        if temp == "hot":
            parts.extend(["steaming", "warm"])
        elif temp == "iced":
            parts.extend(["cold", "chilled"])
        elif temp == "blended":
            parts.extend(["frozen", "blended beverage"])

    caffeine = float(row.get("caffeine_mg", 0) or 0)
    caffeine_info: list[str] = []
    if caffeine == 0:
        caffeine_info.extend(["no caffeine", "caffeine free", "decaf"])
    elif caffeine <= 50:
        caffeine_info.extend(["low caffeine", "mild caffeine"])
    elif caffeine <= 150:
        caffeine_info.extend(["medium caffeine", "regular caffeine"])
    else:
        caffeine_info.extend(["high caffeine", "strong caffeine", "extra caffeine"])

    nutrition: list[str] = []
    calories = float(row.get("calories", 0) or 0)
    sugar = float(row.get("sugar_g", 0) or 0)
    protein = float(row.get("protein_g", 0) or 0)

    if calories > 0:
        nutrition.append(f"{int(calories)} calories")
        if calories <= 50:
            nutrition.append("low calorie")
        elif calories <= 200:
            nutrition.append("moderate calorie")

    if sugar > 0:
        nutrition.append(f"{int(sugar)}g sugar")
        if sugar <= 10:
            nutrition.append("low sugar")
        elif sugar <= 30:
            nutrition.append("moderate sugar")

    if protein > 0:
        nutrition.append(f"{int(protein)}g protein")

    dietary: list[str] = []
    if bool(row.get("is_vegan", False)):
        dietary.extend(["vegan", "plant-based", "no animal products"])

    if not bool(row.get("contains_dairy", False)):
        dietary.extend(["dairy-free", "non-dairy", "no dairy"])
        lower_name = name.lower()
        if "oat" in lower_name or "almond" in lower_name or "coconut" in lower_name:
            dietary.append("plant milk")

    if bool(row.get("contains_nuts", False)):
        dietary.append("contains nuts")

    if caffeine_info:
        parts.append(", ".join(caffeine_info))
    if nutrition:
        parts.append(", ".join(nutrition))
    if dietary:
        parts.append(", ".join(dietary))

    price = float(row.get("price", 0) or 0)
    if price > 0:
        parts.append(f"${price:.2f}")
        if price <= 3.0:
            parts.append("affordable")
        elif price <= 5.0:
            parts.append("moderate price")

    return ". ".join(parts)


def compute_constraint_score(product_row: pd.Series, constraints: Dict[str, Any]) -> float:
    """Compute constraint matching score (0-1)."""
    score = 0.0
    total_weight = 0.0

    if constraints.get("category"):
        if product_row["category"] == constraints["category"]:
            score += 1.0 * 0.3
        total_weight += 0.3

    if constraints.get("temperature"):
        if product_row["temperature"] == constraints["temperature"]:
            score += 1.0 * 0.25
        total_weight += 0.25

    if constraints.get("caffeine_level"):
        from src.config import CAFFEINE_LEVELS

        level = constraints["caffeine_level"]
        if level in CAFFEINE_LEVELS:
            min_caff, _ = CAFFEINE_LEVELS[level]
            product_caff = float(product_row.get("caffeine_mg", 0) or 0)
            if level == "none" and product_caff == 0:
                score += 1.0 * 0.15
            elif level == "low" and 1 <= product_caff <= 50:
                score += 1.0 * 0.15
            elif level == "medium" and 51 <= product_caff <= 150:
                score += 1.0 * 0.15
            elif level == "high" and product_caff >= 151:
                score += 1.0 * 0.15
        total_weight += 0.15

    if constraints.get("dairy_free") is True:
        if not bool(product_row.get("contains_dairy", False)):
            score += 1.0 * 0.1
        total_weight += 0.1

    if constraints.get("vegan") is True:
        if bool(product_row.get("is_vegan", False)):
            score += 1.0 * 0.1
        total_weight += 0.1

    max_calories = constraints.get("max_calories")
    if max_calories is not None:
        product_calories = float(product_row.get("calories", 0) or 0)
        if max_calories <= 0:
            if product_calories <= 0:
                score += 0.05
        elif product_calories <= max_calories:
            ratio = product_calories / max_calories
            score += (1.0 - ratio * 0.5) * 0.05
        total_weight += 0.05

    max_sugar = constraints.get("max_sugar")
    if max_sugar is not None:
        product_sugar = float(product_row.get("sugar_g", 0) or 0)
        if max_sugar <= 0:
            if product_sugar <= 0:
                score += 0.05
        elif product_sugar <= max_sugar:
            ratio = product_sugar / max_sugar
            score += (1.0 - ratio * 0.5) * 0.05
        total_weight += 0.05

    if total_weight > 0:
        return score / total_weight

    return 0.5


def _is_cache_valid(cache_data: Any, current_product_ids: List[str]) -> bool:
    if not isinstance(cache_data, dict):
        return False
    if "embeddings" not in cache_data or "product_ids" not in cache_data:
        return False

    cached_ids = cache_data["product_ids"]
    embeddings = cache_data["embeddings"]

    if not isinstance(cached_ids, list) or cached_ids != current_product_ids:
        return False

    if not isinstance(embeddings, np.ndarray):
        return False

    return embeddings.shape[0] == len(current_product_ids)


def get_product_embeddings(products_df: pd.DataFrame, force_recompute: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Get embeddings for all products, using cache if available.

    If online AI mode is unavailable, returns an empty embedding matrix and ID list.
    """
    cache_path = PRODUCT_EMBEDDINGS_CACHE
    current_product_ids = products_df["product_id"].tolist()

    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as file_obj:
                cache_data = pickle.load(file_obj)

            if _is_cache_valid(cache_data, current_product_ids):
                print(f"Loading product embeddings from cache: {cache_path}")
                return cache_data["embeddings"], cache_data["product_ids"]

            print("Embedding cache mismatch detected, recomputing embeddings...")
        except Exception:
            print("Failed to read embedding cache, recomputing embeddings...")

    if not client:
        print("Online AI mode unavailable. Using lexical ranking mode.")
        return np.empty((0, 0), dtype=float), []

    print("Computing product embeddings...")

    product_texts = products_df.apply(build_product_text, axis=1).tolist()

    batch_size = 100
    all_embeddings: list[list[float]] = []

    for start in range(0, len(product_texts), batch_size):
        batch_texts = product_texts[start : start + batch_size]
        batch_no = start // batch_size + 1
        batch_total = (len(product_texts) + batch_size - 1) // batch_size
        print(f"  Processing batch {batch_no}/{batch_total}")

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        all_embeddings.extend([item.embedding for item in response.data])

    embeddings = np.array(all_embeddings)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, "wb") as file_obj:
        pickle.dump({"embeddings": embeddings, "product_ids": current_product_ids}, file_obj)

    print(f"Cached embeddings to {cache_path}")
    return embeddings, current_product_ids


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _lexical_similarity_scores(query_variations: List[str], candidate_texts: List[str]) -> np.ndarray:
    query_token_sets = [_tokenize(query) for query in query_variations if query.strip()]
    if not query_token_sets:
        return np.zeros(len(candidate_texts), dtype=float)

    scores: list[float] = []
    for candidate in candidate_texts:
        candidate_tokens = _tokenize(candidate)
        if not candidate_tokens:
            scores.append(0.0)
            continue

        best = 0.0
        for query_tokens in query_token_sets:
            if not query_tokens:
                continue
            union_size = len(candidate_tokens | query_tokens)
            if union_size == 0:
                continue
            jaccard = len(candidate_tokens & query_tokens) / union_size
            if jaccard > best:
                best = jaccard
        scores.append(best)

    return np.array(scores, dtype=float)


def _rank_with_scores(candidate_ids: List[str], scores: np.ndarray, top_k: int) -> List[str]:
    sorted_indices = np.argsort(scores)[::-1]
    ranked_ids = [candidate_ids[i] for i in sorted_indices[:top_k]]

    if len(ranked_ids) < top_k:
        remaining = [pid for pid in candidate_ids if pid not in ranked_ids]
        ranked_ids.extend(remaining[: top_k - len(ranked_ids)])

    return ranked_ids


def _offline_rank(
    query_variations: List[str],
    candidate_products: pd.DataFrame,
    constraints: Dict[str, Any] | None,
    top_k: int,
) -> List[str]:
    candidate_ids = candidate_products["product_id"].tolist()
    if not candidate_ids:
        return []

    candidate_texts = candidate_products.apply(build_product_text, axis=1).tolist()
    lexical_scores = _lexical_similarity_scores(query_variations, candidate_texts)

    if constraints:
        constraint_scores = np.array(
            [compute_constraint_score(row, constraints) for _, row in candidate_products.iterrows()],
            dtype=float,
        )
        scores = EMBEDDING_WEIGHT * lexical_scores + CONSTRAINT_WEIGHT * constraint_scores
    else:
        scores = lexical_scores

    return _rank_with_scores(candidate_ids, scores, top_k)


def rank_products(
    query_text: str,
    candidate_products: pd.DataFrame,
    product_embeddings: np.ndarray,
    product_ids: List[str],
    constraints: Dict[str, Any] | None = None,
    top_k: int = TOP_K_PRODUCTS,
) -> List[str]:
    """Rank candidate products using hybrid scoring.

    Uses embedding similarity when available, otherwise falls back to lexical ranking.
    """
    query_variations = expand_query(query_text)
    candidate_ids = candidate_products["product_id"].tolist()

    if not candidate_ids:
        return []

    if not client or product_embeddings.size == 0 or not product_ids:
        return _offline_rank(query_variations, candidate_products, constraints, top_k)

    try:
        query_embeddings: list[np.ndarray] = []
        for variation in query_variations:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=[variation])
            query_embeddings.append(np.array(response.data[0].embedding))

        if not query_embeddings:
            return _offline_rank(query_variations, candidate_products, constraints, top_k)

        query_embedding = np.mean(query_embeddings, axis=0)

        product_index = {pid: idx for idx, pid in enumerate(product_ids)}
        candidate_ids_with_embeddings: list[str] = []
        candidate_indices: list[int] = []
        candidate_rows: list[pd.Series] = []

        for _, row in candidate_products.iterrows():
            pid = row["product_id"]
            emb_idx = product_index.get(pid)
            if emb_idx is None:
                continue
            candidate_ids_with_embeddings.append(pid)
            candidate_indices.append(emb_idx)
            candidate_rows.append(row)

        if not candidate_indices:
            return _offline_rank(query_variations, candidate_products, constraints, top_k)

        candidate_embeddings = product_embeddings[candidate_indices]

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
        )
        embedding_similarities = np.dot(candidate_norms, query_norm)
        embedding_similarities = (embedding_similarities + 1) / 2

        if constraints:
            constraint_scores = np.array(
                [compute_constraint_score(row, constraints) for row in candidate_rows], dtype=float
            )
            hybrid_scores = EMBEDDING_WEIGHT * embedding_similarities + CONSTRAINT_WEIGHT * constraint_scores
        else:
            hybrid_scores = embedding_similarities

        ranked_ids = _rank_with_scores(candidate_ids_with_embeddings, hybrid_scores, top_k)

        if len(ranked_ids) < top_k:
            remaining = [pid for pid in candidate_ids if pid not in ranked_ids]
            ranked_ids.extend(remaining[: top_k - len(ranked_ids)])

        return ranked_ids

    except Exception:
        return _offline_rank(query_variations, candidate_products, constraints, top_k)
