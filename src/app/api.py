"""FastAPI service for online recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import EXPLANATION_TOP_N, HAS_OPENAI_API, MODEL_PRESET, TOP_K_PRODUCTS
from src.core.recommendation_engine import RecommendationEngine


class RecommendRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K_PRODUCTS, ge=1, le=100)
    include_explanations: bool = False
    explanation_top_n: int = Field(default=EXPLANATION_TOP_N, ge=1, le=20)


class BatchItem(BaseModel):
    query_id: Optional[str] = None
    query_text: str = Field(..., min_length=1)


class BatchRecommendRequest(BaseModel):
    items: List[BatchItem]
    top_k: int = Field(default=TOP_K_PRODUCTS, ge=1, le=100)
    include_explanations: bool = False
    explanation_top_n: int = Field(default=EXPLANATION_TOP_N, ge=1, le=20)


app = FastAPI(
    title="Beverage Recommendation Service",
    version="1.0.0",
    description="Production-style API for beverage query understanding and product ranking.",
)
FRONTEND_DIR = Path(__file__).resolve().parent / "demo"
app.mount("/demo", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="demo")

_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine.from_config()
    return _engine


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo/")


@app.get("/health")
def health() -> Dict[str, Any]:
    engine = get_engine()
    return {
        "status": "ok",
        "mode": "online" if HAS_OPENAI_API else "offline",
        "model_preset": MODEL_PRESET,
        "product_count": int(len(engine.products_df)),
    }


@app.get("/v1/config")
def read_config() -> Dict[str, Any]:
    return {
        "mode": "online" if HAS_OPENAI_API else "offline",
        "model_preset": MODEL_PRESET,
        "top_k_default": TOP_K_PRODUCTS,
        "explanation_top_n_default": EXPLANATION_TOP_N,
    }


@app.post("/v1/recommend")
def recommend(request: RecommendRequest) -> Dict[str, Any]:
    try:
        result = get_engine().recommend(
            query_text=request.query_text,
            top_k=request.top_k,
            include_explanations=request.include_explanations,
            explanation_top_n=request.explanation_top_n,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"recommendation_failed: {exc}") from exc


@app.post("/v1/recommend/batch")
def recommend_batch(request: BatchRecommendRequest) -> Dict[str, Any]:
    if not request.items:
        raise HTTPException(status_code=400, detail="items must not be empty")

    try:
        payload_items = [item.model_dump() for item in request.items]
        results = get_engine().recommend_batch(
            queries=payload_items,
            top_k=request.top_k,
            include_explanations=request.include_explanations,
            explanation_top_n=request.explanation_top_n,
        )
        return {"count": len(results), "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"batch_recommendation_failed: {exc}") from exc
