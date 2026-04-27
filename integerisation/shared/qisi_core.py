from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .utils import generate_halton_sequence, scramble_halton_cranley_patterson


def perform_qisi(
    df: pd.DataFrame,
    target_count: int,
    stratify_cols: List[str],
    weight_col: str,
    integer_weight_col: str,
    seed: int,
) -> pd.DataFrame:
    df = df.copy()

    if target_count <= 0:
        raise ValueError(f"target_count must be a positive integer, got: {target_count}")
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in dataframe")

    missing_stratify = [col for col in stratify_cols if col not in df.columns]
    if missing_stratify:
        raise ValueError(f"Stratification columns not found in dataframe: {missing_stratify}")

    weights = df[weight_col].to_numpy()
    if np.isnan(weights).any():
        raise ValueError("Weight column contains NaN values")
    if (weights < 0).any():
        raise ValueError("Weight column contains negative values")

    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Total weight <= 0, cannot integerise")

    sort_columns = stratify_cols + [weight_col]
    sort_ascending = [True] * len(stratify_cols) + [False]
    df = df.sort_values(by=sort_columns, ascending=sort_ascending).reset_index(drop=True)

    weights = df[weight_col].to_numpy()
    intervals = np.cumsum(weights) / weights.sum()
    intervals[-1] = 1.0

    halton_raw = generate_halton_sequence(target_count, base=2)
    halton_scrambled = scramble_halton_cranley_patterson(halton_raw, seed)
    selected_indices = np.searchsorted(intervals, halton_scrambled, side="left")
    selected_indices = np.clip(selected_indices, 0, len(df) - 1)

    df[integer_weight_col] = np.bincount(selected_indices, minlength=len(df)).astype(int)

    total_integer_weight = int(df[integer_weight_col].sum())
    if total_integer_weight != target_count:
        raise RuntimeError(
            f"QISI validation failed: integer weight sum ({total_integer_weight}) != target count ({target_count})"
        )

    return df
