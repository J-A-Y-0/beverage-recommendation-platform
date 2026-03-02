"""OpenAI client factory with optional OpenAI-compatible endpoint support."""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from src.config import HAS_OPENAI_API, OPENAI_API_KEY, OPENAI_BASE_URL


def get_openai_client() -> Optional[OpenAI]:
    """Return an initialized OpenAI client when online mode is enabled."""
    if not HAS_OPENAI_API:
        return None

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return OpenAI(**kwargs)
