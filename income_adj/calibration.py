"""
Time-space income calibration.

Formula: syn_income = (hybrid_annual_wage / S_origin) * S_target
- Urban: S = provincial average wage by industry x ownership type (private / non-private)
- Rural:  S = prefecture-level rural per capita disposable income
"""

import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import PathConfig, ProvinceConfig, DIRECT_MUNICIPALITY_CODES as MUNICIPALITY_CODES, SURVEY_YEAR


FINAL_CSV_COLUMNS = [
    "ID",
    "target_geo",
    "U_R",
    "integer_weight",
    "syn_income",
    "C_INDUSTRY",
    "C_OCCUPATION",
    "C_EDU_WORKER",
    "C_SEX",
    "AGE_BAND",
]


# ============================================================================
# Required Columns
# ============================================================================


REQUIRED_INPUT_COLS = [
    "ID", "coun", "integer_weight", "stratum_id", "U_R", "hybrid_annual_wage", "company_ownership", "full_no_full",
]

PASSTHROUGH_OUTPUT_COLS = [
    "C_INDUSTRY","C_OCCUPATION","C_EDU_WORKER","AGE_BAND","C_SEX",]

TARGET_INCOME_COL = "syn_income"
RATIO_COL = "ratio"
TARGET_YEAR = 2020


def _target_scale_year() -> int:
    return TARGET_YEAR


def _target_scale_col() -> str:
    return f"S_target_{_target_scale_year()}"
def _final_output_cols() -> List[str]:
    return [
        "ID", "target_geo", "U_R", "origin_geo", "province_code", "integer_weight",
        "hybrid_annual_wage", "S_origin", _target_scale_col(),
        RATIO_COL, TARGET_INCOME_COL, 
        *PASSTHROUGH_OUTPUT_COLS,
        "company_ownership", "full_no_full",
    ]


def _require_u_r_values(series: pd.Series) -> pd.Series:
    """Require deterministic urban/rural values in the fixed 2020 workflow."""
    u_r_values = series.astype("string")
    invalid = ~u_r_values.isin(["1", "2"])
    if invalid.any():
        sample = u_r_values[invalid].drop_duplicates().tolist()[:10]
        raise ValueError(f"Invalid U_R values found: {sample}")
    return u_r_values



# ============================================================================
# Stratum Parsing
# ============================================================================

def parse_stratum_id_vectorized(stratum_ids: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse stratum_id -> (target_geo, target_cc)"""
    split = stratum_ids.str.split("|", n=1, expand=True)
    target_geo = split[0]
    target_cc = split[1] if 1 in split.columns else pd.Series(None, index=stratum_ids.index, dtype=object)
    return target_geo, target_cc


def correct_geo_codes_vectorized(geo_codes: pd.Series) -> pd.Series:
    """Apply geo code corrections"""
    corrected = geo_codes.astype(str).copy()
    for old_code, new_code in ProvinceConfig.GEO_CODE_CORRECTIONS.items():
        corrected = corrected.replace(old_code, new_code)
    return corrected


# ============================================================================
# Geographic Lookups  
# ============================================================================



def _map_district_to_city(
    geo_str: pd.Series,
    county_to_city: Dict[str, str],
) -> pd.Series:
    """Shared: municipality→province_code, district→county_to_city with [:4]+'00' fallback."""
    province_codes = geo_str.str[:2]
    city_codes = pd.Series(index=geo_str.index, dtype=object)

    is_municipality = province_codes.isin(MUNICIPALITY_CODES)
    city_codes.loc[is_municipality] = province_codes.loc[is_municipality]

    non_muni_mask = ~is_municipality
    if non_muni_mask.any():
        non_muni_geo = geo_str.loc[non_muni_mask]
        mapped = non_muni_geo.map(county_to_city)
        unmapped = mapped.isna()
        if unmapped.any():
            mapped.loc[unmapped] = non_muni_geo.loc[unmapped].str[:4] + "00"
        city_codes.loc[non_muni_geo.index] = mapped

    return city_codes


def lookup_city_code_vectorized(
    geo_codes: pd.Series,
    county_to_city: Dict[str, str]
) -> pd.Series:
    """Map geo codes to city codes using scenario-specific province granularity."""
    geo_str = geo_codes.astype(str).map(ProvinceConfig.map_city_scale_geo_code)
    province_codes = geo_str.str[:2]

    city_codes = pd.Series(index=geo_str.index, dtype=object)

    is_municipality = province_codes.isin(MUNICIPALITY_CODES)
    city_codes[is_municipality] = province_codes[is_municipality]

    non_muni_mask = ~is_municipality
    if non_muni_mask.any():
        non_muni_geo = geo_str[non_muni_mask]
        non_muni_province = province_codes[non_muni_mask]

        city_level_mask = non_muni_province.map(
            lambda code: ProvinceConfig.is_city_granularity(code)
        )
        if city_level_mask.any():
            city_codes.loc[city_level_mask.index[city_level_mask]] = non_muni_geo.loc[city_level_mask]

        district_mask = ~city_level_mask
        if district_mask.any():
            district_geo = non_muni_geo.loc[district_mask]
            city_codes.loc[district_geo.index] = _map_district_to_city(district_geo, county_to_city).loc[district_geo.index]

    return city_codes


def lookup_origin_city_from_coun_vectorized(
    geo_codes: pd.Series,
    county_to_city: Dict[str, str]
) -> pd.Series:
    """Map origin coun codes to prefecture-level city codes."""
    return _map_district_to_city(geo_codes.astype(str), county_to_city)

def lookup_province_code_vectorized(geo_codes: pd.Series) -> pd.Series:
    """Map county codes to province codes (first 2 digits)"""
    return geo_codes.astype(str).str[:2]


def _map_city_scale_with_raw_rural_fallback(
    city_codes: pd.Series,
    income_dict: Dict[str, float],
    cc_val: str
) -> pd.Series:
    """
    Map city scales. For rural (U_R=2), fallback to RAW_{city_code} when city key missing.
    """
    mapped = city_codes.map(income_dict)
    if cc_val == "2":
        missing = mapped.isna()
        if missing.any():
            raw_keys = "RAW_" + city_codes.loc[missing].astype(str)
            mapped.loc[missing] = raw_keys.map(income_dict)
    return mapped


def _get_scale_dict(
    city_scale_lookup: Dict[str, Dict[str, float]],
    key: str,
    context: str = "",
) -> Dict[str, float]:
    """Fail fast when a U_R-year key is missing from the lookup."""
    if key not in city_scale_lookup:
        suffix = f" ({context})" if context else ""
        raise ValueError(f"[missing lookup] Scale lookup key not found: {key}{suffix}")
    return city_scale_lookup[key]


def _assert_no_missing_scale(
    df: pd.DataFrame,
    mask: pd.Series,
    scale_col: str,
    diag_cols: List[str],
    label: str,
) -> None:
    """Raise if any rows in *mask* are still NaN in *scale_col*."""
    missing = mask & df[scale_col].isna()
    if missing.any():
        sample = (
            df.loc[missing, diag_cols]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{label}: {missing.sum()} rows have no city-level data (province fallback not allowed). sample={sample}"
        )


def _map_urban_industry_scale(
    df: pd.DataFrame,
    mask: pd.Series,
    province_col: str,
    year: int,
    urban_industry_scale_lookup: Dict[int, Dict[Tuple[str, int, str], float]],
    scale_col: str,
    label: str,
) -> None:
    lookup = urban_industry_scale_lookup.get(int(year))
    if lookup is None:
        raise ValueError(f"Urban industry wage data missing for year: {year}")

    rows = df.loc[mask, [province_col, "C_INDUSTRY", "company_ownership"]].copy()
    rows["province_code"] = rows[province_col].astype(str).str[:2]
    rows["C_INDUSTRY"] = pd.to_numeric(rows["C_INDUSTRY"], errors="raise").astype(int)
    rows["company_ownership"] = rows["company_ownership"].astype(str).str.strip()
    keys = list(zip(rows["province_code"], rows["C_INDUSTRY"], rows["company_ownership"]))
    values = [lookup.get(key) for key in keys]
    df.loc[mask, scale_col] = values

    missing = mask & df[scale_col].isna()
    if missing.any():
        sample = (
            df.loc[missing, [province_col, "C_INDUSTRY", "company_ownership"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"{label}: {missing.sum()} rows have no urban industry wage match. sample={sample}")


# ============================================================================
# Calibration: City-level Disposable Income
# ============================================================================

def compute_city_disposable_scales_for_province(
    df: pd.DataFrame,
    county_to_city: Dict[str, str],
    rural_city_scale_lookup: Dict[str, Dict[str, float]],
    urban_industry_scale_lookup: Dict[int, Dict[Tuple[str, int, str], float]],
) -> pd.DataFrame:
    """
    Compute S_origin (fixed survey year) and S_target (TARGET_YEAR) for each row,
    then derive ratio = hybrid_annual_wage / S_origin.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    
    df = df.copy()
    target_year = _target_scale_year()
    target_col = _target_scale_col()

    # Parse stratum_id -> target_geo, U_R
    target_geo, _ = parse_stratum_id_vectorized(df["stratum_id"])
    df["target_geo"] = target_geo
    df["U_R"] = _require_u_r_values(df["U_R"])
    df["origin_geo"] = correct_geo_codes_vectorized(df["coun"])
    
    # Map to city codes
    df["origin_city"] = lookup_origin_city_from_coun_vectorized(df["origin_geo"], county_to_city)
    df["target_city"] = lookup_city_code_vectorized(df["target_geo"], county_to_city)
    
    # Province codes
    df["origin_province"] = df["origin_geo"].astype(str).str[:2]
    df["target_province"] = df["target_geo"].astype(str).str[:2]
    df["province_code"] = df["target_province"]
    df["is_employed"] = True

    # Income: annual basis (fixed upstream schema)
    df["hybrid_annual_wage"] = df["hybrid_annual_wage"].astype(float)
    
    # Initialize scale columns
    df["S_origin"] = np.nan
    df[target_col] = np.nan
    
    df["origin_year_for_scale"] = SURVEY_YEAR
    
    urban_mask = df["U_R"] == "1"
    if urban_mask.any():
        origin_years = (
            df.loc[urban_mask, "origin_year_for_scale"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        for year in sorted(origin_years):
            year_mask = urban_mask & (df["origin_year_for_scale"] == year)
            if year_mask.any():
                _map_urban_industry_scale(
                    df=df,
                    mask=year_mask,
                    province_col="origin_province",
                    year=int(year),
                    urban_industry_scale_lookup=urban_industry_scale_lookup,
                    scale_col="S_origin",
                    label=f"[S_origin missing] U_R=1, year={year}",
                )
        _map_urban_industry_scale(
            df=df,
            mask=urban_mask,
            province_col="target_province",
            year=target_year,
            urban_industry_scale_lookup=urban_industry_scale_lookup,
            scale_col=target_col,
            label=f"[S_target missing] U_R=1, year={target_year}",
        )

    rural_mask = df["U_R"] == "2"
    if rural_mask.any():
        origin_years = (
            df.loc[rural_mask, "origin_year_for_scale"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        for year in sorted(origin_years):
            sub_mask = rural_mask & (df["origin_year_for_scale"] == year)
            if not sub_mask.any():
                continue
            income_dict = _get_scale_dict(rural_city_scale_lookup, f"2_{int(year)}", "S_origin")
            df.loc[sub_mask, "S_origin"] = _map_city_scale_with_raw_rural_fallback(
                df.loc[sub_mask, "origin_city"], income_dict, "2"
            )
            _assert_no_missing_scale(
                df, sub_mask, "S_origin",
                ["origin_city", "origin_province", "U_R", "origin_year_for_scale"],
                f"[S_origin missing] U_R=2, year={year}",
            )

        income_dict = _get_scale_dict(rural_city_scale_lookup, f"2_{target_year}", "S_target")
        df.loc[rural_mask, target_col] = _map_city_scale_with_raw_rural_fallback(
            df.loc[rural_mask, "target_city"], income_dict, "2"
        )
        _assert_no_missing_scale(
            df, rural_mask, target_col,
            ["target_city", "target_province", "U_R"],
            f"[S_target missing] U_R=2, year={target_year}",
        )

    # Compute ratio
    valid_mask = (
        df["is_employed"] &
        df["hybrid_annual_wage"].notna() &
        (df["hybrid_annual_wage"] > 0) &
        df["S_origin"].notna() &
        (df["S_origin"] > 0)
    )

    df[RATIO_COL] = np.nan
    df.loc[valid_mask, RATIO_COL] = (
        df.loc[valid_mask, "hybrid_annual_wage"] /
        df.loc[valid_mask, "S_origin"]
    )
    
    return df


# ============================================================================
# Main Streaming Calibration
# ============================================================================

def run_streaming_calibration(
    county_to_city: Dict[str, str],
    rural_city_scale_lookup: Dict[str, Dict[str, float]],
    urban_industry_scale_lookup: Dict[int, Dict[Tuple[str, int, str], float]],
    adjusted_dir: Path,
    final_output_dir: Path,
) -> None:
    """
    Run streaming calibration province by province.
    syn_income = (hybrid_annual_wage / S_origin) * S_target_{TARGET_YEAR}

    Writes per-province:
    - adjusted_dir/adjusted_<prov>.parquet  (full columns for downstream validation)
    - final_output_dir/province_<prov>.csv  (reduced columns, utf-8-sig)
    """
    target_col = _target_scale_col()
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    read_cols = []
    for col in REQUIRED_INPUT_COLS + PASSTHROUGH_OUTPUT_COLS:
        if col not in read_cols:
            read_cols.append(col)

    for prov_code in PathConfig.list_premerged_provinces():
        path = PathConfig.get_premerged_path(prov_code)
        df = pd.read_parquet(path, columns=read_cols)

        result_df = compute_city_disposable_scales_for_province(
            df=df,
            county_to_city=county_to_city,
            rural_city_scale_lookup=rural_city_scale_lookup,
            urban_industry_scale_lookup=urban_industry_scale_lookup,
        )
        del df

        if len(result_df) == 0:
            gc.collect()
            continue

        result_df[TARGET_INCOME_COL] = np.nan
        valid_mask = (
            result_df["is_employed"] &
            result_df[RATIO_COL].notna() & (result_df[RATIO_COL] > 0) &
            result_df[target_col].notna() & (result_df[target_col] > 0)
        )
        result_df.loc[valid_mask, TARGET_INCOME_COL] = (
            result_df.loc[valid_mask, RATIO_COL] * result_df.loc[valid_mask, target_col]
        )

        parquet_cols = [c for c in _final_output_cols() if c in result_df.columns]
        result_df[parquet_cols].to_parquet(
            adjusted_dir / f"adjusted_{prov_code}.parquet", index=False, compression="snappy"
        )

        result_df[FINAL_CSV_COLUMNS].to_csv(
            final_output_dir / f"province_{prov_code}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del result_df
        gc.collect()
