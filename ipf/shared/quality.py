from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config


@dataclass
class CategoryAccumulator:
    targets: List[float] = field(default_factory=list)
    fitted: List[float] = field(default_factory=list)
    rae_values: List[float] = field(default_factory=list)


def _compute_pearson_r(targets: List[float], fitted: List[float]) -> float:
    targets_arr = np.array(targets, dtype=float)
    fitted_arr = np.array(fitted, dtype=float)

    if len(targets_arr) <= 1:
        return np.nan
    if np.std(targets_arr) == 0 or np.std(fitted_arr) == 0:
        return np.nan
    return float(np.corrcoef(targets_arr, fitted_arr)[0, 1])


def _compute_nrmse_percent(targets: List[float], fitted: List[float]) -> float:
    targets_arr = np.array(targets, dtype=float)
    fitted_arr = np.array(fitted, dtype=float)
    if len(targets_arr) == 0:
        return np.nan

    obs_range = targets_arr.max() - targets_arr.min()
    if obs_range <= 0:
        return np.nan

    rmse = np.sqrt(np.mean((fitted_arr - targets_arr) ** 2))
    return float(rmse / obs_range * 100.0)


def _format_quantiles(values: List[float]) -> str:
    if not values:
        return ""
    quantiles = np.quantile(np.array(values, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return "/".join(f"{value:.4f}" for value in quantiles)


def _parse_weight_file_strata() -> set[str]:
    strata = set()
    for path in config.WEIGHTS_DIR.glob("*.parquet"):
        strata.add(path.stem.replace("_", "|", 1))
    return strata


def _load_runtime_df() -> pd.DataFrame:
    runtime_df = pd.read_csv(config.RUNTIME_METADATA_PATH, dtype={"stratum_id": str})
    if runtime_df.empty:
        return runtime_df
    runtime_df["stratum_id"] = runtime_df["stratum_id"].astype(str)
    return runtime_df


def _compute_geo_metrics(
    targets_dict: Dict[str, Dict],
    fitted_values: Dict[str, Dict],
) -> Dict[str, float]:
    all_targets: List[float] = []
    all_fitted: List[float] = []
    rae_values: List[float] = []

    for variable, categories in targets_dict.items():
        for category, target in categories.items():
            target_value = float(target)
            fitted_value = float(fitted_values.get(variable, {}).get(category, 0.0))
            all_targets.append(target_value)
            all_fitted.append(fitted_value)
            if target_value > 0:
                rae_values.append(abs(fitted_value - target_value) / target_value * 100.0)

    return {
        "rae% (min/q1/median/q3/max)": _format_quantiles(rae_values),
        "nrmse%": _compute_nrmse_percent(all_targets, all_fitted),
        "pearson_r": _compute_pearson_r(all_targets, all_fitted),
    }


def generate_quality_reports() -> Tuple[pd.DataFrame, pd.DataFrame]:
    microdata = pd.read_parquet(config.MICRODATA_PATH)
    macro_master = pd.read_parquet(config.MACRO_MASTER_PATH)
    total_pop_df = pd.read_parquet(config.MARGIN_TOTAL_POP_PATH)
    zero_total_pop_strata = config.get_zero_total_pop_strata(total_pop_df)
    runtime_df = _load_runtime_df()

    if runtime_df.empty:
        empty_geo = pd.DataFrame(
            columns=["stratum_id", "rae% (min/q1/median/q3/max)", "nrmse%", "pearson_r", "iterations", "max_weight"]
        )
        empty_var = pd.DataFrame(
            columns=["variable", "category", "rae% (min/q1/median/q3/max)", "nrmse%", "pearson_r"]
        )
        empty_geo.to_csv(config.QUALITY_BY_GEO_PATH, index=False, encoding="utf-8-sig")
        empty_var.to_csv(config.QUALITY_BY_VARIABLE_CATEGORY_PATH, index=False, encoding="utf-8-sig")
        return empty_geo, empty_var

    available_weight_ids = _parse_weight_file_strata()
    runtime_df = runtime_df[runtime_df["stratum_id"].isin(available_weight_ids)].copy()
    runtime_df = runtime_df[~runtime_df["stratum_id"].astype(str).isin(zero_total_pop_strata)].copy()
    runtime_df = runtime_df.sort_values("stratum_id").reset_index(drop=True)

    geo_rows: List[Dict[str, float]] = []
    category_accumulators: Dict[Tuple[str, str], CategoryAccumulator] = {}

    for row in runtime_df.itertuples(index=False):
        stratum_id = str(row.stratum_id)
        weights_path = config.WEIGHTS_DIR / f"{config.safe_stratum_id(stratum_id)}.parquet"
        weights_df = pd.read_parquet(weights_path)
        constraints = macro_master[macro_master["stratum_id"] == stratum_id].copy()
        if constraints.empty:
            continue

        merged = weights_df[[config.MICRO_ID_COLUMN, "weight"]].merge(
            microdata,
            on=config.MICRO_ID_COLUMN,
            how="left",
        )

        targets_dict: Dict[str, Dict] = {}
        fitted_values: Dict[str, Dict] = {}

        for variable in constraints["variable"].unique():
            if variable not in merged.columns:
                continue

            var_constraints = constraints[constraints["variable"] == variable]
            fitted_by_category = merged.groupby(variable, dropna=False)["weight"].sum()
            targets_dict[variable] = {}
            fitted_values[variable] = {}

            for constraint_row in var_constraints.itertuples(index=False):
                category = constraint_row.category
                target = float(constraint_row.count)
                fitted = float(fitted_by_category.get(category, 0.0))
                targets_dict[variable][category] = target
                fitted_values[variable][category] = fitted

                acc = category_accumulators.setdefault((variable, str(category)), CategoryAccumulator())
                acc.targets.append(target)
                acc.fitted.append(fitted)
                if target > 0:
                    acc.rae_values.append(abs(fitted - target) / target * 100.0)

        geo_metrics = _compute_geo_metrics(
            targets_dict=targets_dict,
            fitted_values=fitted_values,
        )
        geo_rows.append(
            {
                "stratum_id": stratum_id,
                "rae% (min/q1/median/q3/max)": geo_metrics["rae% (min/q1/median/q3/max)"],
                "nrmse%": geo_metrics["nrmse%"],
                "pearson_r": geo_metrics["pearson_r"],
                "iterations": int(row.iterations),
                "max_weight": float(row.max_weight),
            }
        )

    geo_df = pd.DataFrame(
        geo_rows,
        columns=["stratum_id", "rae% (min/q1/median/q3/max)", "nrmse%", "pearson_r", "iterations", "max_weight"],
    )
    if not geo_df.empty:
        geo_df = geo_df.sort_values("stratum_id").reset_index(drop=True)

    variable_rows = []
    for (variable, category), acc in sorted(category_accumulators.items()):
        variable_rows.append(
            {
                "variable": variable,
                "category": category,
                "rae% (min/q1/median/q3/max)": _format_quantiles(acc.rae_values),
                "nrmse%": _compute_nrmse_percent(acc.targets, acc.fitted),
                "pearson_r": _compute_pearson_r(acc.targets, acc.fitted),
            }
        )

    variable_df = pd.DataFrame(
        variable_rows,
        columns=["variable", "category", "rae% (min/q1/median/q3/max)", "nrmse%", "pearson_r"],
    )
    if not variable_df.empty:
        variable_df = variable_df.sort_values(["variable", "category"]).reset_index(drop=True)

    geo_df.to_csv(config.QUALITY_BY_GEO_PATH, index=False, encoding="utf-8-sig")
    variable_df.to_csv(config.QUALITY_BY_VARIABLE_CATEGORY_PATH, index=False, encoding="utf-8-sig")
    return geo_df, variable_df
