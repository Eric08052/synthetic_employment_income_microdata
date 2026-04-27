"""Shared utility functions for validation modules."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import REGION_CLASSIFICATION


def convert_for_json(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def get_region(province_code: str) -> str:
    """Get region (East/Central/West) from province code."""
    return REGION_CLASSIFICATION.get(str(province_code).zfill(2)[:2], "Unknown")


def compute_shared_lims(
    frames: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    *,
    pad: Optional[float] = None,
    pad_ratio: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Compute shared axis limits across multiple DataFrames.

    Padding modes (keyword-only, mutually exclusive):
    - pad: absolute padding (e.g. 0.1)
    - pad_ratio: proportional padding (e.g. 0.02 = 2% of span)
    - neither specified: defaults to pad=0.1
    """
    mins, maxs = [], []
    for df in frames:
        if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
            continue
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = x.notna() & y.notna()
        if not valid.any():
            continue
        mins.append(float(min(x[valid].min(), y[valid].min())))
        maxs.append(float(max(x[valid].max(), y[valid].max())))
    if not mins or not maxs:
        return None
    lo, hi = min(mins), max(maxs)
    if pad is not None:
        actual_pad = pad
    elif pad_ratio is not None:
        span = hi - lo
        if span <= 0:
            span = max(1.0, abs(lo) * 0.1)
        actual_pad = span * pad_ratio
    else:
        actual_pad = 0.1
    if hi <= lo:
        hi = lo + 1e-6
    return (lo - actual_pad, hi + actual_pad)


def compute_ordinal_metrics(merged: pd.DataFrame, x_col: str, y_col: str) -> Dict:
    """Compute Spearman rank correlation between two columns."""
    spearman_r, _ = scipy_stats.spearmanr(merged[x_col], merged[y_col])
    return {"spearman_rho": float(spearman_r)}
