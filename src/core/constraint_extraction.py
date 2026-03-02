"""Extract structured constraints from query text."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional

from src.config import EXTRACTION_MODEL, MAX_RETRIES, RETRY_DELAY
from src.openai_client import get_openai_client

client = get_openai_client()

DEFAULT_CONSTRAINTS: Dict[str, Any] = {
    "category": None,
    "temperature": None,
    "caffeine_level": None,
    "max_calories": None,
    "max_sugar": None,
    "max_price": None,
    "dairy_free": None,
    "vegan": None,
}

VALID_CATEGORIES = {"brewed", "cold_brew", "espresso", "frappuccino", "refresher", "tea", None}
VALID_TEMPERATURES = {"hot", "iced", "blended", None}
VALID_CAFFEINE = {"none", "low", "medium", "high", None}


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _extract_numeric(text: str, patterns: list[str]) -> Optional[float]:
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                values.append(float(match.group(1)))
            except (ValueError, TypeError, IndexError):
                continue

    if not values:
        return None

    # Conservative for "max" constraints: the strictest value is safest.
    return min(values)


def _detect_category(text: str) -> Optional[str]:
    rules = [
        ("cold_brew", ["cold brew", "nitro cold brew", "iced coffee"]),
        ("frappuccino", ["frappuccino", "frozen drink", "blended drink", "frozen coffee"]),
        (
            "refresher",
            ["refresher", "refreshing and fruity", "fruity refresher", "fruit refresher"],
        ),
        (
            "tea",
            ["tea", "chai", "green tea", "herbal tea", "matcha", "earl grey", "tea latte"],
        ),
        (
            "espresso",
            [
                "espresso",
                "latte",
                "cappuccino",
                "americano",
                "mocha",
                "macchiato",
                "flat white",
                "cortado",
            ],
        ),
        (
            "brewed",
            [
                "drip coffee",
                "house coffee",
                "black coffee",
                "plain coffee",
                "regular coffee",
                "brewed coffee",
                "coffee",
            ],
        ),
    ]

    for category, terms in rules:
        if _contains_any(text, terms):
            return category

    return None


def _detect_temperature(text: str) -> Optional[str]:
    if _contains_any(text, ["blended", "frozen", "frappuccino"]):
        return "blended"

    hot_terms = ["hot", "steaming", "warm", "steamed"]
    iced_terms = ["iced", "cold", "on ice", "chilled", "icy"]

    hot_pos = min([text.find(term) for term in hot_terms if term in text], default=-1)
    iced_pos = min([text.find(term) for term in iced_terms if term in text], default=-1)

    if hot_pos == -1 and iced_pos == -1:
        return None
    if hot_pos == -1:
        return "iced"
    if iced_pos == -1:
        return "hot"

    return "hot" if hot_pos < iced_pos else "iced"


def _detect_caffeine_level(text: str) -> Optional[str]:
    levels = {
        "none": ["no caffeine", "caffeine free", "decaf", "decaffeinated", "without caffeine"],
        "low": ["low caffeine", "mild", "not too much caffeine", "light caffeine", "little caffeine"],
        "medium": ["regular caffeine", "medium caffeine", "normal caffeine", "standard caffeine"],
        "high": ["high caffeine", "strong", "lots of caffeine", "need the caffeine", "extra caffeine"],
    }

    matches: list[tuple[int, str]] = []
    for label, keywords in levels.items():
        for keyword in keywords:
            pos = text.find(keyword)
            if pos >= 0:
                matches.append((pos, label))
                break

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _detect_bool_constraint(text: str, true_terms: list[str], false_terms: list[str]) -> Optional[bool]:
    has_true = _contains_any(text, true_terms)
    has_false = _contains_any(text, false_terms)

    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False
    return None


def _rule_based_extract(query_text: str) -> Dict[str, Any]:
    text = query_text.lower()

    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints["category"] = _detect_category(text)
    constraints["temperature"] = _detect_temperature(text)
    constraints["caffeine_level"] = _detect_caffeine_level(text)

    constraints["max_calories"] = _extract_numeric(
        text,
        [
            r"(?:under|less than|below|no more than|max(?:imum)?|at most)\s*(\d+(?:\.\d+)?)\s*(?:calories?|cals?|kcal)",
            r"(\d+(?:\.\d+)?)\s*(?:calories?|cals?|kcal)\s*(?:or less|or under|max)",
        ],
    )

    constraints["max_sugar"] = _extract_numeric(
        text,
        [
            r"(?:under|less than|below|no more than|max(?:imum)?|at most)\s*(\d+(?:\.\d+)?)\s*(?:g|grams?)?\s*sugar",
            r"(\d+(?:\.\d+)?)\s*(?:g|grams?)\s*(?:of\s*)?sugar\s*(?:or less|or under|max)?",
        ],
    )

    constraints["max_price"] = _extract_numeric(
        text,
        [
            r"(?:under|less than|below|cheaper than|no more than|max(?:imum)?|at most)\s*\$\s*(\d+(?:\.\d+)?)",
            r"(?:under|less than|below|cheaper than|no more than|max(?:imum)?|at most)\s*(\d+(?:\.\d+)?)\s*(?:dollars?|bucks)",
            r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|or under|max)",
            r"(\d+(?:\.\d+)?)\s*(?:dollars?|bucks)\s*(?:or less|or under|max)",
        ],
    )

    constraints["dairy_free"] = _detect_bool_constraint(
        text,
        true_terms=[
            "dairy free",
            "no dairy",
            "non-dairy",
            "without milk",
            "no regular milk",
            "plant based",
            "oat milk",
            "almond milk",
            "coconut milk",
        ],
        false_terms=["with milk", "contains dairy", "dairy is fine", "regular milk"],
    )

    constraints["vegan"] = _detect_bool_constraint(
        text,
        true_terms=["vegan", "no animal stuff", "plant based", "fully vegan", "vegan friendly", "no animal products"],
        false_terms=["not vegan", "non-vegan", "animal products are fine"],
    )

    return _normalize_constraints(constraints)


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    return None


def _normalize_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        numbers = re.findall(r"\d+\.?\d*", value)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
    return None


def _normalize_constraints(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized = DEFAULT_CONSTRAINTS.copy()

    category = raw.get("category")
    if isinstance(category, str):
        category = category.strip().lower()
    normalized["category"] = category if category in VALID_CATEGORIES else None

    temperature = raw.get("temperature")
    if isinstance(temperature, str):
        temperature = temperature.strip().lower()
    normalized["temperature"] = temperature if temperature in VALID_TEMPERATURES else None

    caffeine_level = raw.get("caffeine_level")
    if isinstance(caffeine_level, str):
        caffeine_level = caffeine_level.strip().lower()
    normalized["caffeine_level"] = caffeine_level if caffeine_level in VALID_CAFFEINE else None

    normalized["max_calories"] = _normalize_numeric(raw.get("max_calories"))
    normalized["max_sugar"] = _normalize_numeric(raw.get("max_sugar"))
    normalized["max_price"] = _normalize_numeric(raw.get("max_price"))

    normalized["dairy_free"] = _normalize_bool(raw.get("dairy_free"))
    normalized["vegan"] = _normalize_bool(raw.get("vegan"))

    return normalized


def _llm_extract(query_text: str) -> Dict[str, Any]:
    if not client:
        raise RuntimeError("OpenAI client not configured")

    prompt = f"""You extract structured constraints from drink-order queries.

Query: \"{query_text}\"

Return ONLY valid JSON with this schema:
{{
  "category": null,
  "temperature": null,
  "caffeine_level": null,
  "max_calories": null,
  "max_sugar": null,
  "max_price": null,
  "dairy_free": null,
  "vegan": null
}}

Rules:
- category must be one of: brewed, cold_brew, espresso, frappuccino, refresher, tea, null
- temperature must be one of: hot, iced, blended, null
- caffeine_level must be one of: none, low, medium, high, null
- Use null for omitted constraints
- For numeric fields, extract a number only (no units)
- dairy_free and vegan should be true / false / null
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise JSON extraction assistant. Return only a JSON object.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")
            return _normalize_constraints(parsed)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise

    raise RuntimeError("Failed to extract constraints")


def extract_constraints(query_text: str) -> Dict[str, Any]:
    """Extract constraints from a natural-language query.

    Uses the LLM when configured; otherwise falls back to rule-based extraction.
    """
    if client:
        try:
            return _llm_extract(query_text)
        except Exception:
            # Keep pipeline robust when API is unavailable or transiently failing.
            return _rule_based_extract(query_text)

    return _rule_based_extract(query_text)
