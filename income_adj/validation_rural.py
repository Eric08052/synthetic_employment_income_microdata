"""
Rural Validation

County-level rural disposable income validation:
- Compare county-level synthetic rural employment income (U_R='2') against external rural disposable income.
- Only run for district-level provinces (skip city-granularity provinces).

Key Outputs:
1. Validation metrics in JSON
2. Log-log scatter for baseline and adjusted runs
"""

import gc
import json
import logging
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

from config import (
    PathConfig,
    ProvinceConfig,
)
from data_loader import load_total_pop_lookup, load_external_rural_income_data
from validation_utils import convert_for_json, get_region, compute_shared_lims, compute_ordinal_metrics

from plot_style import (
    plot_validation_rural_scatter,
)
from main import apply_path_overrides

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

def get_rural_province_whitelist() -> set:
    """District-level provinces for rural validation."""
    granularity = ProvinceConfig.PROVINCE_GRANULARITY
    no_track = ProvinceConfig.NO_TRACK_PROVINCE_CODES
    all_codes = set(ProvinceConfig.PROVINCE_CODE_TO_NAME.keys())
    return {
        code for code in all_codes
        if code not in no_track
        and granularity.get(code, "district") == "district"
    }


# ============================================================================
# Adjusted Data Loading & Aggregation
# ============================================================================

def load_and_aggregate_adjusted(
    adjusted_dir: Path,
    income_col: str = "syn_income",
    province_whitelist: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load adjusted parquets and aggregate by county (rural only, district-level provinces).
    """
    parquet_files = sorted(adjusted_dir.glob("adjusted_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No adjusted_XX.parquet files found in: {adjusted_dir}")

    if province_whitelist is None:
        province_whitelist = get_rural_province_whitelist()

    county_agg = {}
    for pf in parquet_files:
        province_code = pf.stem.replace("adjusted_", "", 1)
        if province_code not in province_whitelist:
            continue

        cols_to_read = ["target_geo", "integer_weight", "U_R", income_col]
        df = pd.read_parquet(pf, columns=cols_to_read)
        df["county_code"] = df["target_geo"].astype("string")

        # Filters: rural, valid income, valid county code
        mask = (df["U_R"] == "2") & df[income_col].notna() & (df[income_col] > 0) & df["county_code"].notna()
        df = df[mask]

        if len(df) == 0:
            continue

        for county_code, group in df.groupby("county_code"):
            if county_code not in county_agg:
                county_agg[county_code] = {"sum_income": 0.0, "sum_weight": 0.0}
            weights = group["integer_weight"].fillna(0)
            incomes = group[income_col].fillna(0)
            county_agg[county_code]["sum_income"] += (weights * incomes).sum()
            county_agg[county_code]["sum_weight"] += weights.sum()

        del df

    gc.collect()

    if not county_agg:
        raise ValueError("No valid rural data after filtering")

    rows = [
        {"county_code": code, "mean_income": v["sum_income"] / v["sum_weight"],
         "sum_income": v["sum_income"], "sum_weight": v["sum_weight"]}
        for code, v in county_agg.items() if v["sum_weight"] > 0
    ]

    result_df = pd.DataFrame(rows)
    result_df["province_code"] = result_df["county_code"].str[:2]
    result_df["region"] = result_df["province_code"].apply(get_region)

    return result_df


def apply_total_pop_denominator(
    county_income: pd.DataFrame,
    total_pop_lookup: Dict[Tuple[str, str, str], float],
) -> pd.DataFrame:
    """Replace denominator with total population. Strict: no fallback."""
    df = county_income.copy()
    df["total_pop_county"] = [
        total_pop_lookup.get((str(p), str(c), "2"))
        for p, c in zip(df["province_code"], df["county_code"])
    ]

    missing = df["total_pop_county"].isna()
    if missing.any():
        sample = df.loc[missing, ["province_code", "county_code"]].drop_duplicates().head(15).to_dict("records")
        raise ValueError(f"Missing total_pop (strict). sample={sample}")

    df["mean_income"] = df["sum_income"] / df["total_pop_county"]
    return df


# ============================================================================
# Validation Metrics
# ============================================================================

def compute_regional_metrics(
    merged: pd.DataFrame,
    income_col: str = "mean_income",
    income_col_ext: str = "rural_income",
) -> Dict:
    results = {"by_region": []}

    for region in ["East", "Central", "West"]:
        region_data = merged[merged["region"] == region]

        if len(region_data) < 3:
            results["by_region"].append({
                "region": region, "n": len(region_data), "status": "insufficient_data",
            })
            continue

        x = region_data[income_col].values
        y = region_data[income_col_ext].values
        valid_mask = np.isfinite(x) & np.isfinite(y)

        if valid_mask.sum() < 3:
            results["by_region"].append({
                "region": region, "n": int(valid_mask.sum()), "status": "insufficient_valid",
            })
            continue

        rho, p_val = scipy_stats.spearmanr(x[valid_mask], y[valid_mask])
        results["by_region"].append({
            "region": region, "n": int(valid_mask.sum()),
            "spearman_rho": rho, "p_value": p_val,
        })

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_income_scatter(
    merged: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
    income_col: str = "log_sim",
    income_col_ext: str = "log_ext",
    shared_lims: Optional[Tuple[float, float]] = None,
):
    plot_validation_rural_scatter(
        merged,
        output_path,
        income_col=income_col,
        income_col_ext=income_col_ext,
        shared_lims=shared_lims,
        axis_label_mode="log",
    )


# ============================================================================
# Main Validation Function
# ============================================================================

def run_validation_rural(
    adjusted_dir: Path,
    output_dir: Path,
    is_baseline: bool = False,
    income_col: str = "syn_income",
) -> dict:
    """
    Run rural validation: county-level rural disposable income.
    Returns dict with validation results; "_merged" key holds the merged DataFrame.
    """
    province_whitelist = get_rural_province_whitelist()

    results = {
        "data_scenario": "2020",
        "province_whitelist": sorted(province_whitelist),
        "denominator_mode": "total_pop",
    }

    # Load external data
    rural_income_df = load_external_rural_income_data(province_whitelist=province_whitelist)

    # Load synthetic data
    actual_income_col = "hybrid_annual_wage" if is_baseline else income_col
    county_income = load_and_aggregate_adjusted(
        adjusted_dir=adjusted_dir,
        income_col=actual_income_col,
        province_whitelist=province_whitelist,
    )

    # Apply total_pop denominator
    total_pop_lookup = load_total_pop_lookup(province_whitelist=province_whitelist)
    county_income = apply_total_pop_denominator(county_income, total_pop_lookup)

    # Unmatched diagnostics
    # Merge
    merged = county_income.merge(rural_income_df, on="county_code", how="inner", suffixes=("", "_ext"))
    if "region_ext" in merged.columns:
        merged["region"] = merged["region_ext"]
    merged["log_sim"] = np.log(merged["mean_income"])
    merged["log_ext"] = np.log(merged["rural_income"])

    results["n_matched_counties"] = len(merged)
    results["filtered_counties"] = len(rural_income_df)

    # Metrics
    results["ordinal"] = compute_ordinal_metrics(merged, "mean_income", "rural_income")
    results["regional"] = compute_regional_metrics(merged)

    # Scatter plot
    file_suffix = "baseline" if is_baseline else "adjusted"
    plot_income_scatter(
        merged, output_dir / "log_scatter.png",
        title_suffix=f"({file_suffix}, total_pop)",
    )

    # Save JSON
    json_path = output_dir / f"validation_results_{file_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(convert_for_json(results), f, indent=2, ensure_ascii=False)

    results["status"] = "success"
    results["_merged"] = merged
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    output_dir = PathConfig.VALIDATION_RURAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    adjusted_dir = PathConfig.get_adjusted_dir()
    if not adjusted_dir.exists():
        logger.error(f"Adjusted directory not found: {adjusted_dir}")
        sys.exit(1)

    province_whitelist = sorted(get_rural_province_whitelist())
    logger.info(f"Rural whitelist (2020): {province_whitelist}")

    merged_frames = {}

    for mode_name in ["baseline", "adjusted"]:
        is_baseline = (mode_name == "baseline")
        run_dir = (
            PathConfig.VALIDATION_RURAL_BASELINE_OUTPUT_DIR
            if is_baseline
            else PathConfig.VALIDATION_RURAL_ADJUSTED_OUTPUT_DIR
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_validation_rural(
            adjusted_dir=adjusted_dir,
            output_dir=run_dir,
            is_baseline=is_baseline,
        )

        merged_frames[mode_name] = result.pop("_merged")
        rho = result["ordinal"].get("spearman_rho", "N/A")
        rho_str = f"{rho:.4f}" if isinstance(rho, float) else str(rho)
        logger.info(f"{mode_name}: matched={result['n_matched_counties']} spearman={rho_str}")

    # Re-draw scatter plots with shared axis limits
    if len(merged_frames) == 2:
        shared_lims = compute_shared_lims(
            list(merged_frames.values()),
            x_col="log_ext", y_col="log_sim",
            pad=0.1,
        )
        if shared_lims is not None:
            for mode_name, mdf in merged_frames.items():
                run_dir = (
                    PathConfig.VALIDATION_RURAL_BASELINE_OUTPUT_DIR
                    if mode_name == "baseline"
                    else PathConfig.VALIDATION_RURAL_ADJUSTED_OUTPUT_DIR
                )
                plot_income_scatter(
                    mdf,
                    run_dir / "log_scatter.png",
                    title_suffix=f"({mode_name}, total_pop)",
                    shared_lims=shared_lims,
                )


if __name__ == "__main__":
    cli_args = argparse.ArgumentParser(description="Run new rural validation with optional path overrides.")
    cli_args.add_argument("--premerged-dir", type=Path, default=None)
    cli_args.add_argument("--output-dir", type=Path, default=None)
    parsed_args = cli_args.parse_args()
    apply_path_overrides(premerged_dir=parsed_args.premerged_dir, output_dir=parsed_args.output_dir)
    main()
