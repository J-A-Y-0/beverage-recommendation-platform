"""Filter products based on extracted constraints."""
import pandas as pd
from typing import Dict, Any
from src.config import CAFFEINE_LEVELS


def filter_products_strict(products_df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """Apply constraints without fallback."""
    if not constraints:
        return products_df.copy()

    filtered = products_df.copy()
    
    if constraints.get("category"):
        filtered = filtered[filtered["category"] == constraints["category"]]
    
    if constraints.get("temperature"):
        filtered = filtered[filtered["temperature"] == constraints["temperature"]]
    
    if constraints.get("max_calories") is not None:
        filtered = filtered[filtered["calories"] <= constraints["max_calories"]]
    
    if constraints.get("max_sugar") is not None:
        filtered = filtered[filtered["sugar_g"] <= constraints["max_sugar"]]
    
    if constraints.get("max_price") is not None:
        filtered = filtered[filtered["price"] <= constraints["max_price"]]
    
    if constraints.get("dairy_free") is True:
        filtered = filtered[~filtered["contains_dairy"]]
    
    if constraints.get("vegan") is True:
        filtered = filtered[filtered["is_vegan"]]
    
    if constraints.get("caffeine_level"):
        level = constraints["caffeine_level"]
        if level in CAFFEINE_LEVELS:
            min_caff, max_caff = CAFFEINE_LEVELS[level]
            if level == "none":
                filtered = filtered[filtered["caffeine_mg"] == min_caff]
            elif level == "high":
                filtered = filtered[filtered["caffeine_mg"] >= min_caff]
            else:
                filtered = filtered[
                    (filtered["caffeine_mg"] >= min_caff) & 
                    (filtered["caffeine_mg"] <= max_caff)
                ]
    
    return filtered


def filter_products(products_df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """Filter products based on constraints. Applies fallback relaxation if empty."""
    filtered = filter_products_strict(products_df, constraints)
    
    if len(filtered) == 0:
        filtered = apply_fallback_relaxation(products_df, constraints)
    
    return filtered


def apply_fallback_relaxation(products_df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
    """Relax constraints if initial filtering returns empty set."""
    if not constraints:
        return products_df.copy()

    relaxed_constraints = constraints.copy()
    
    if relaxed_constraints.get("max_price") is not None:
        test_constraints = {k: v for k, v in relaxed_constraints.items() if k != "max_price"}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("max_sugar") is not None:
        test_constraints = {k: v for k, v in relaxed_constraints.items() if k != "max_sugar"}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("max_calories") is not None:
        test_constraints = {k: v for k, v in relaxed_constraints.items() if k != "max_calories"}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("temperature"):
        test_constraints = {k: v for k, v in relaxed_constraints.items() if k != "temperature"}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("caffeine_level"):
        test_constraints = {k: v for k, v in relaxed_constraints.items() if k != "caffeine_level"}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("dairy_free") is True or relaxed_constraints.get("vegan") is True:
        test_constraints = {k: v for k, v in relaxed_constraints.items() 
                          if k not in ["dairy_free", "vegan"]}
        filtered = filter_products_strict(products_df, test_constraints)
        if len(filtered) > 0:
            return filtered
    
    if relaxed_constraints.get("category"):
        filtered = products_df[products_df["category"] == relaxed_constraints["category"]]
        if len(filtered) > 0:
            return filtered
    
    return products_df
