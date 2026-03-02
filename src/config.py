"""Configuration settings."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _resolve_data_path(env_var: str, private_filename: str, sample_filename: str) -> str:
    """Resolve data file path with env override and sample fallback."""
    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    private_path = BASE_DIR / private_filename
    if private_path.exists():
        return str(private_path)

    return str(DATA_DIR / sample_filename)


MODEL_PRESETS = {
    # Fastest online setup with lower cost.
    "fast": {
        "extraction_model": "gpt-4o-mini",
        "expansion_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "use_query_expansion": False,
    },
    # Balanced default for public onboarding.
    "balanced": {
        "extraction_model": "gpt-4o-mini",
        "expansion_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "use_query_expansion": True,
    },
    # Higher-quality online ranking.
    "quality": {
        "extraction_model": "gpt-4o",
        "expansion_model": "gpt-4o",
        "embedding_model": "text-embedding-3-large",
        "use_query_expansion": True,
    },
    # Fully local fallback mode.
    "offline": {
        "extraction_model": "gpt-4o-mini",
        "expansion_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "use_query_expansion": False,
    },
}

MODEL_PRESET = os.getenv("MODEL_PRESET", "balanced").strip().lower()
if MODEL_PRESET not in MODEL_PRESETS:
    MODEL_PRESET = "balanced"
ACTIVE_MODEL_PRESET = MODEL_PRESETS[MODEL_PRESET]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
FORCE_OFFLINE_MODE = _env_bool("FORCE_OFFLINE_MODE", MODEL_PRESET == "offline")
HAS_OPENAI_API = bool(OPENAI_API_KEY) and not FORCE_OFFLINE_MODE

EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", ACTIVE_MODEL_PRESET["extraction_model"])
EXPANSION_MODEL = os.getenv("EXPANSION_MODEL", ACTIVE_MODEL_PRESET["expansion_model"])
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", ACTIVE_MODEL_PRESET["embedding_model"])
USE_QUERY_EXPANSION = _env_bool(
    "USE_QUERY_EXPANSION", bool(ACTIVE_MODEL_PRESET["use_query_expansion"])
)

# Backward compatibility for legacy imports.
CHAT_MODEL = EXTRACTION_MODEL

PRODUCTS_CSV = _resolve_data_path("PRODUCTS_CSV", "products.csv", "sample_products.csv")
QUERIES_TRAIN_CSV = _resolve_data_path(
    "QUERIES_TRAIN_CSV", "queries_train.csv", "sample_queries_train.csv"
)
QUERIES_TEST_CSV = _resolve_data_path(
    "QUERIES_TEST_CSV", "queries_test.csv", "sample_queries_test.csv"
)
SUBMISSION_CSV = os.getenv("SUBMISSION_CSV", str(BASE_DIR / "outputs/submission.csv"))
PRODUCT_EMBEDDINGS_CACHE = os.getenv(
    "PRODUCT_EMBEDDINGS_CACHE", str(BASE_DIR / "cache/product_embeddings.pkl")
)
WRITE_EXPLANATIONS = _env_bool("WRITE_EXPLANATIONS", False)
EXPLANATIONS_JSON = os.getenv(
    "EXPLANATIONS_JSON", str(BASE_DIR / "outputs/recommendation_explanations.json")
)
EXPLANATION_TOP_N = max(1, _env_int("EXPLANATION_TOP_N", 5))

MAX_RETRIES = 3
RETRY_DELAY = 1.0

TOP_K_PRODUCTS = 25
EMBEDDING_WEIGHT = 0.7
CONSTRAINT_WEIGHT = 0.3

CAFFEINE_LEVELS = {
    "none": (0, 0),
    "low": (1, 50),
    "medium": (51, 150),
    "high": (151, float("inf")),
}
