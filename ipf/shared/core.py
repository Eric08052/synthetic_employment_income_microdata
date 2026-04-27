from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_constraint_matrix(
    seed_df: pd.DataFrame,
    constraints: pd.DataFrame,
    target_columns: List[str],
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    targets_dict: Dict[str, Dict] = {}
    indicators_dict: Dict[str, Dict] = {}

    for var in target_columns:
        if var not in seed_df.columns:
            continue

        var_constraints = constraints[constraints["variable"] == var]
        if var_constraints.empty:
            continue

        targets_dict[var] = {}
        indicators_dict[var] = {}

        for _, row in var_constraints.iterrows():
            category = row["category"]
            targets_dict[var][category] = row["count"]
            indicators_dict[var][category] = (seed_df[var] == category).values

    return targets_dict, indicators_dict


def ipf_iterate(
    weights: np.ndarray,
    targets_dict: Dict[str, Dict],
    indicators_dict: Dict[str, Dict],
    cap_lower: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int]:
    current_weights = weights.copy().astype(np.float64)

    for var, categories in targets_dict.items():
        for category, target in categories.items():
            if target == 0:
                mask = indicators_dict[var][category]
                current_weights[mask] = 0.0

    for iteration in range(1, max_iter + 1):
        previous_weights = current_weights.copy()

        for var, categories in targets_dict.items():
            for category, target in categories.items():
                if target == 0:
                    continue

                mask = indicators_dict[var][category]
                current_sum = np.sum(current_weights[mask])
                if current_sum > 0:
                    current_weights[mask] *= target / current_sum

        current_weights = np.maximum(current_weights, cap_lower)
        max_weight_change = np.max(np.abs(current_weights - previous_weights))
        if max_weight_change < tol:
            return current_weights, iteration

    return current_weights, max_iter
