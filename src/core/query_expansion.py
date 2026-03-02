"""Query expansion for better semantic matching."""

from __future__ import annotations

import json
import re
from typing import List

from src.config import EXPANSION_MODEL, USE_QUERY_EXPANSION
from src.openai_client import get_openai_client

client = get_openai_client()


def _rule_based_variations(query_text: str) -> List[str]:
    """Generate deterministic local rewrites for offline mode."""
    variations: list[str] = []
    text = query_text

    replacements = [
        (r"\bcoffee\b", "brewed coffee"),
        (r"\bblack coffee\b", "plain brewed coffee"),
        (r"\bno dairy\b", "dairy free"),
        (r"\bwithout milk\b", "non-dairy"),
        (r"\bvegan\b", "plant-based"),
        (r"\bunder\b", "less than"),
        (r"\biced\b", "cold"),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, text, flags=re.IGNORECASE):
            candidate = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if candidate != text and candidate not in variations:
                variations.append(candidate)
        if len(variations) >= 2:
            break

    return variations


def _sanitize_variations(query_text: str, variations: List[str]) -> List[str]:
    deduped: list[str] = []
    seen = {query_text.strip().lower()}

    for item in variations:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue

        key = normalized.lower()
        if key in seen:
            continue

        seen.add(key)
        deduped.append(normalized)

        if len(deduped) >= 2:
            break

    return [query_text] + deduped


def expand_query(query_text: str) -> List[str]:
    """Generate semantic query variations.

    Returns up to three entries: original query + up to two variations.
    """
    if not USE_QUERY_EXPANSION:
        return [query_text]

    if not client:
        return _sanitize_variations(query_text, _rule_based_variations(query_text))

    try:
        prompt = f"""Generate 2 semantic variations of this customer query with the same meaning.

Original query: \"{query_text}\"

Requirements:
1. Keep all numeric and dietary constraints unchanged
2. Use alternate wording for drink type or intent
3. Do not add new requirements

Return only JSON in this shape:
{{"variations": ["variation 1", "variation 2"]}}
"""

        response = client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate safe paraphrases for search queries. Return only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        variations = parsed.get("variations", []) if isinstance(parsed, dict) else []

        if not isinstance(variations, list):
            variations = []

        if not variations:
            variations = _rule_based_variations(query_text)

        return _sanitize_variations(query_text, variations)

    except Exception:
        return _sanitize_variations(query_text, _rule_based_variations(query_text))
