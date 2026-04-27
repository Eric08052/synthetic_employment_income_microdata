"""
Urban Validation
City-level wage validation (synthetic vs external urban non-private workers)

Validation Scope:
- Urban employment (U_R = '1')
- Non-private sector (company_ownership = '1')

Key Outputs:
1. Validation metrics in JSON
2. Log-log scatter for baseline and adjusted runs
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

from config import (
    PathConfig,
)
from data_loader import parse_geo_cc_from_stratum_id, load_geo_mapping, resolve_city_code_from_geo, load_external_city_wage
from validation_utils import convert_for_json, get_region, compute_shared_lims, compute_ordinal_metrics
from plot_style import (
    plot_validation_urban_scatter,
)
from main import apply_path_overrides

# ============================================================================
# City Code Matching
# ============================================================================


_COUNTY_TO_CITY, _, _ = load_geo_mapping()


def _geo_to_city(geo_code: str) -> Optional[str]:
    """Resolve geo code to city code."""
    return resolve_city_code_from_geo(geo_code, _COUNTY_TO_CITY)


def _stratum_to_city(stratum_id: str) -> Optional[str]:
    """Resolve stratum_id to city code."""
    geo_part, _ = parse_geo_cc_from_stratum_id(stratum_id)
    if geo_part is None:
        return None
    return resolve_city_code_from_geo(geo_part, _COUNTY_TO_CITY)


def get_province_code(city_code: str) -> str:
    """Extract province code from city code"""
    if pd.isna(city_code):
        return None
    return str(city_code)[:2]


# ============================================================================
# Streaming urban aggregation
# ============================================================================

def _load_and_aggregate_urban_core(
    data_dir: Path,
    file_pattern: str,
    required_cols: list,
    income_col: str,
    geo_mapper_fn,
    geo_source_col: str,
    extra_filter_fn=None,
    post_agg_fn=None,
) -> pd.DataFrame:
    """Core streaming: load parquet files, filter urban non-private workers, aggregate by city."""
    parquet_files = sorted(data_dir.glob(file_pattern))
    city_aggregates = []

    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=required_cols)
        df = df[df["U_R"].astype(str).str.strip().eq("1")].copy()
        if df.empty:
            continue

        mask = df["company_ownership"].astype(str).str.strip().eq("1")
        if extra_filter_fn is not None:
            mask &= extra_filter_fn(df)
        df = df[mask]
        if df.empty:
            continue

        df["city_code"] = df[geo_source_col].apply(geo_mapper_fn)
        df = df[df["city_code"].notna()]
        if df.empty:
            continue

        df["income_w"] = df[income_col] * df["integer_weight"]
        agg = df.groupby("city_code").agg({
            "integer_weight": "sum",
            "income_w": "sum",
            income_col: "count"
        }).reset_index()
        agg.columns = ["city_code", "sum_weight", "income_sum_w", "n_raw"]
        city_aggregates.append(agg)

    combined = pd.concat(city_aggregates, ignore_index=True)
    final_agg = combined.groupby("city_code").agg({
        "sum_weight": "sum",
        "income_sum_w": "sum",
        "n_raw": "sum"
    }).reset_index()

    final_agg["mean_wage_sim"] = final_agg["income_sum_w"] / final_agg["sum_weight"]
    final_agg["province_code"] = final_agg["city_code"].apply(get_province_code)

    if post_agg_fn is not None:
        post_agg_fn(final_agg)
    return final_agg


def load_and_aggregate_streaming(adjusted_dir: Path) -> pd.DataFrame:
    """Adjusted: stream adjusted_*.parquet, aggregate syn_income by city."""
    required_cols = [
        "target_geo", "U_R", "company_ownership",
        "integer_weight", "syn_income",
    ]
    result = _load_and_aggregate_urban_core(
        data_dir=adjusted_dir,
        file_pattern="adjusted_*.parquet",
        required_cols=required_cols,
        income_col="syn_income",
        geo_mapper_fn=_geo_to_city,
        geo_source_col="target_geo",
        extra_filter_fn=lambda df: df["syn_income"].notna() & (df["syn_income"] > 0),
    )
    result["n_eff"] = result["sum_weight"]
    return result


def load_and_aggregate_baseline_streaming(premerged_dir: Path) -> pd.DataFrame:
    """Baseline: stream province_*.parquet, aggregate hybrid_annual_wage by city."""
    required_cols = [
        "stratum_id", "U_R", "integer_weight",
        "hybrid_annual_wage", "company_ownership",
    ]
    result = _load_and_aggregate_urban_core(
        data_dir=premerged_dir,
        file_pattern="province_*.parquet",
        required_cols=required_cols,
        income_col="hybrid_annual_wage",
        geo_mapper_fn=_stratum_to_city,
        geo_source_col="stratum_id",
        extra_filter_fn=lambda df: df["hybrid_annual_wage"].notna() & (df["hybrid_annual_wage"] > 0),
        post_agg_fn=lambda agg: agg.__setitem__("region", agg["province_code"].apply(get_region)),
    )
    gc.collect()
    return result


# ============================================================================
# Data Merging Functions
# ============================================================================

def merge_sim_ext(
    sim_agg: pd.DataFrame,
    ext_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge simulated city aggregates with external data
    
    Args:
        sim_agg: City-level simulated aggregates
        ext_df: External city wage data
        
    Returns:
        Merged DataFrame with both sim and ext values
    """
    # Extract matching city code from external data
    ext_df = ext_df.copy()
    ext_df["city_code_match"] = ext_df["city_code"].apply(_geo_to_city)
    
    # Prepare for merge
    sim_agg = sim_agg.copy()
    
    # Merge on city code
    merged = sim_agg.merge(
        ext_df[["city_code_match", "avg_wage_annual"]],
        left_on="city_code",
        right_on="city_code_match",
        how="inner"
    )

    # Rename for clarity (now using annual wage)
    merged["mean_wage_ext"] = merged["avg_wage_annual"]
    
    # Add log values
    merged["log_sim"] = np.log(merged["mean_wage_sim"])
    merged["log_ext"] = np.log(merged["mean_wage_ext"])
    
    # Add region
    merged["region"] = merged["province_code"].apply(get_region)
    
    return merged


# ============================================================================
# Linear Structure Metrics (Log Scale)
# ============================================================================

def compute_log_regression(merged: pd.DataFrame) -> Dict:
    """
    Compute log-log regression: log(sim) = α + β·log(ext) + ε
    
    Interpretation:
    - α ≠ 0: Overall multiplicative/level bias
    - β < 1: Regression-to-mean (gaps compressed)
    - β > 1: Gaps amplified
    
    Args:
        merged: Merged DataFrame with log_sim and log_ext
        
    Returns:
        Dict with regression results
    """
    x = merged["log_ext"].values
    y = merged["log_sim"].values
    
    # OLS regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    
    results = {
        "alpha": float(intercept),
        "beta": float(slope),
        "r_squared": float(r_value ** 2),
        "r": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }
    return results


# ============================================================================
# Absolute Error Metrics
# ============================================================================

def compute_absolute_error_metrics(merged: pd.DataFrame) -> Dict:
    """Compute reported relative absolute error diagnostics."""
    sim = merged["mean_wage_sim"].to_numpy()
    ext = merged["mean_wage_ext"].to_numpy()

    rel_errors = np.abs(sim - ext) / ext

    rae_summary = {
        "min": float(np.min(rel_errors)),
        "q1": float(np.percentile(rel_errors, 25)),
        "median": float(np.median(rel_errors)),
        "q3": float(np.percentile(rel_errors, 75)),
        "max": float(np.max(rel_errors)),
    }

    return {
        "rae_summary": rae_summary,
        "rae_str": (
            f"{rae_summary['min']:.1%}/{rae_summary['q1']:.1%}/"
            f"{rae_summary['median']:.1%}/{rae_summary['q3']:.1%}/{rae_summary['max']:.1%}"
        ),
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_log_log_scatter(
    merged: pd.DataFrame,
    regression: Dict,
    output_path: Path,
    lims: Optional[Tuple[float, float]] = None,
):
    """Plot log-log scatter with regression line"""
    plot_validation_urban_scatter(merged, regression, output_path, lims=lims)


def enforce_shared_loglog_axes_compare_urban(
    baseline_merged: pd.DataFrame,
    adjusted_merged: pd.DataFrame,
    baseline_output_dir: Path,
    adjusted_output_dir: Path,
) -> Tuple[float, float]:
    """
    Re-draw baseline/adjusted log-log scatter with shared axis limits.
    """
    shared_lims = compute_shared_lims(
        [baseline_merged, adjusted_merged],
        x_col="log_ext",
        y_col="log_sim",
        pad=0.1,
    )
    if shared_lims is None:
        raise ValueError("Compare shared-axis redraw failed: unable to compute shared limits.")

    baseline_reg = compute_log_regression(baseline_merged)
    adjusted_reg = compute_log_regression(adjusted_merged)

    plot_log_log_scatter(
        baseline_merged,
        baseline_reg,
        baseline_output_dir / "log_scatter.png",
        lims=shared_lims,
    )
    plot_log_log_scatter(
        adjusted_merged,
        adjusted_reg,
        adjusted_output_dir / "log_scatter.png",
        lims=shared_lims,
    )
    return shared_lims


# ============================================================================
# Main Validation Function
# ============================================================================

def build_validation_results(merged: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, Dict]:
    ordinal_metrics = compute_ordinal_metrics(merged, "mean_wage_sim", "mean_wage_ext")
    ordinal_by_region = []
    for region, region_df in merged.groupby("region", dropna=False):
        if len(region_df) < 2:
            continue
        region_metrics = compute_ordinal_metrics(region_df, "mean_wage_sim", "mean_wage_ext")
        ordinal_by_region.append(
            {
                "region": region,
                "spearman_rho": region_metrics["spearman_rho"],
                "n_cities": int(len(region_df)),
            }
        )
    ordinal_metrics["by_region"] = sorted(ordinal_by_region, key=lambda row: str(row["region"]))
    regression = compute_log_regression(merged)
    absolute_errors = compute_absolute_error_metrics(merged)
    results = {
        "n_matched_cities": int(len(merged)),
        "ordinal": ordinal_metrics,
        "absolute_errors": absolute_errors,
    }
    return results, merged, regression


def run_validation_urban(
    adjusted_dir: Path = None,
    external_wage_path: Path = None,
    output_dir: Path = None,
) -> Tuple[Dict, pd.DataFrame]:
    if adjusted_dir is None:
        adjusted_dir = PathConfig.get_adjusted_dir()
    if external_wage_path is None:
        external_wage_path = PathConfig.VALIDATION_URBAN_CITY_WAGE
    external_wage_path = Path(external_wage_path)
    if output_dir is None:
        output_dir = PathConfig.VALIDATION_URBAN_ADJUSTED_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext_df = load_external_city_wage(external_wage_path)

    sim_agg = load_and_aggregate_streaming(
        adjusted_dir=adjusted_dir,
    )
    if len(sim_agg) == 0:
        raise ValueError("Aggregation result is empty")

    merged = merge_sim_ext(sim_agg, ext_df)
    results, merged, regression = build_validation_results(merged)
    results = finalize_validation_outputs(merged, output_dir, "adjusted", results, regression)
    return results, merged


def finalize_validation_outputs(
    merged: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    results: Dict,
    regression: Dict,
) -> Dict:
    plot_log_log_scatter(merged, regression, output_dir / "log_scatter.png")
    with open(output_dir / f"validation_results_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(convert_for_json(results), f, indent=2, ensure_ascii=False)
    return results


def run_baseline_validation_single(
    premerged_dir: Path,
    ext_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Dict, pd.DataFrame]:
    """Run baseline validation with fixed urban non-private scope."""
    sim_agg = load_and_aggregate_baseline_streaming(
        premerged_dir=premerged_dir,
    )
    if len(sim_agg) == 0:
        raise ValueError("No data after aggregation")

    merged = merge_sim_ext(sim_agg, ext_df)
    results, merged, regression = build_validation_results(merged)
    results = finalize_validation_outputs(merged, output_dir, "baseline", results, regression)
    return results, merged


# ============================================================================
# Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run new urban SAE validation with optional path overrides.")
    parser.add_argument("--premerged-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    apply_path_overrides(premerged_dir=cli_args.premerged_dir, output_dir=cli_args.output_dir)
    premerged_dir = PathConfig.PREMERGED_DIR
    adjusted_dir = PathConfig.get_adjusted_dir()
    external_wage_path = Path(PathConfig.VALIDATION_URBAN_CITY_WAGE)

    compare_output_base = PathConfig.VALIDATION_URBAN_OUTPUT_DIR
    compare_output_base.mkdir(parents=True, exist_ok=True)
    baseline_output_dir = PathConfig.VALIDATION_URBAN_BASELINE_OUTPUT_DIR
    adjusted_output_dir = PathConfig.VALIDATION_URBAN_ADJUSTED_OUTPUT_DIR
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    adjusted_output_dir.mkdir(parents=True, exist_ok=True)

    ext_df = load_external_city_wage(external_wage_path)

    results_baseline, baseline_merged = run_baseline_validation_single(
        premerged_dir=premerged_dir,
        ext_df=ext_df,
        output_dir=baseline_output_dir,
    )

    results_adjusted, adjusted_merged = run_validation_urban(
        adjusted_dir=adjusted_dir,
        external_wage_path=external_wage_path,
        output_dir=adjusted_output_dir,
    )

    enforce_shared_loglog_axes_compare_urban(
        baseline_merged=baseline_merged,
        adjusted_merged=adjusted_merged,
        baseline_output_dir=baseline_output_dir,
        adjusted_output_dir=adjusted_output_dir,
    )
