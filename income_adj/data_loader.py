"""
Income Calibration System (Simplified) - Data Loader Module
Loads geo mapping, calibration scales, and external validation data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from config import PathConfig, DIRECT_MUNICIPALITY_CODES
from validation_utils import get_region

logger = logging.getLogger(__name__)


# ============================================================================
# Geographic Mapping
# ============================================================================

def load_geo_mapping() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load geographic code mappings from JSON

    Returns:
        Tuple of:
        - county_to_city: {county_code: city_code}
        - county_to_province: {county_code: province_code}
        - city_to_province: {city_code: province_code}
    """
    path = PathConfig.GEO_MAPPING_JSON

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    county_to_city = {}
    county_to_province = {}
    city_to_province = {}

    for record in data.values():
        province_code = record["province_code"][:2]
        county_code = record["county_code"]
        city_code = record["city_code"]

        county_to_province[county_code] = province_code
        county_to_city[county_code] = city_code
        city_to_province[city_code] = province_code

    return county_to_city, county_to_province, city_to_province


# ============================================================================
# Geographic Utilities
# ============================================================================


def _normalize_geo_code_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, float):
        text = str(int(value)) if value.is_integer() else str(value).rstrip("0").rstrip(".")
    else:
        text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_geo_code_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_geo_code_value)


def parse_geo_cc_from_stratum_id(stratum_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse fixed-format stratum_id ('geo|U_R') into (geo_code, U_R)."""
    if pd.isna(stratum_id):
        return None, None

    raw = str(stratum_id).strip()
    geo_code, u_r = raw.split("|", 1)
    return geo_code.strip(), u_r.strip()


def resolve_city_code_from_geo(geo_code: str, county_to_city: Dict[str, str]) -> Optional[str]:
    """Map geo code (county/city) to prefecture-level city code key used by urban."""
    if geo_code is None:
        return None

    geo = str(geo_code).strip()
    if not geo:
        return None

    province_code = geo[:2]
    if province_code in DIRECT_MUNICIPALITY_CODES:
        return province_code

    if geo in county_to_city:
        return str(county_to_city[geo])

    if len(geo) == 6 and geo[4:] == "00":
        return geo

    if len(geo) >= 4:
        return geo[:4] + "00"

    return None


# ============================================================================
# Calibration Scale Data
# ============================================================================

URBAN_CITY_DISPOSABLE_REQUIRED_COLS = ["city_code", "可支配收入", "year"]
URBAN_WAGE_REQUIRED_COLS = ["行政区划代码", "年份"]
NONPRIVATE_WAGE_COLS = [
    "农、林、牧、渔业城镇单位就业人员平均工资(元)",
    "采矿业城镇单位就业人员平均工资(元)",
    "制造业城镇单位就业人员平均工资(元)",
    "电力、燃气及水的生产和供应业城镇单位就业人员平均工资(元)",
    "建筑业城镇单位就业人员平均工资(元)",
    "交通运输、仓储和邮政业城镇单位就业人员平均工资(元)",
    "信息传输、计算机服务和软件业城镇单位就业人员平均工资(元)",
    "批发和零售业城镇单位就业人员平均工资(元)",
    "住宿和餐饮业城镇单位就业人员平均工资(元)",
    "金融业城镇单位就业人员平均工资(元)",
    "房地产业城镇单位就业人员平均工资(元)",
    "租赁和商务服务业城镇单位就业人员平均工资(元)",
    "科学研究、技术服务和地质勘查业城镇单位就业人员平均工资(元)",
    "水利、环境和公共设施管理业城镇单位就业人员平均工资(元)",
    "居民服务和其他服务业城镇单位就业人员平均工资(元)",
    "教育城镇单位就业人员平均工资(元)",
    "卫生、社会保障和社会福利业城镇单位就业人员平均工资(元)",
    "文化、体育和娱乐业城镇单位就业人员平均工资(元)",
    "公共管理和社会组织城镇单位就业人员平均工资(元)",
]
PRIVATE_WAGE_COLS = [
    "农、林、牧、渔业城镇私营单位就业人员平均工资(元)",
    "采矿业城镇私营单位就业人员平均工资(元)",
    "制造业城镇私营单位就业人员平均工资(元)",
    "电力、燃气及水的生产和供应业城镇私营单位就业人员平均工资(元)",
    "建筑业城镇私营单位就业人员平均工资(元)",
    "交通运输、仓储和邮政业城镇私营单位就业人员平均工资(元)",
    "信息传输、计算机服务和软件业城镇私营单位就业人员平均工资(元)",
    "批发和零售业城镇私营单位就业人员平均工资(元)",
    "住宿和餐饮业城镇私营单位就业人员平均工资(元)",
    "金融业城镇私营单位就业人员平均工资(元)",
    "房地产业城镇私营单位就业人员平均工资(元)",
    "租赁和商务服务业城镇私营单位就业人员平均工资(元)",
    "科学研究、技术服务和地质勘查业城镇私营单位就业人员平均工资(元)",
    "水利、环境和公共设施管理业城镇私营单位就业人员平均工资(元)",
    "居民服务和其他服务业城镇私营单位就业人员平均工资(元)",
    "教育城镇私营单位就业人员平均工资(元)",
    "卫生、社会保障和社会福利业城镇私营单位就业人员平均工资(元)",
    "文化、体育和娱乐业城镇私营单位就业人员平均工资(元)",
    "公共管理和社会组织城镇私营单位就业人员平均工资(元)",
]


def load_city_disposable_income_triple() -> Dict[str, Dict[str, float]]:
    """
    Load city-level disposable income by urban/rural and year.
    """
    path = PathConfig.URBAN_CITY_DISPOSABLE_INCOME

    result = {}

    sheet_mapping = {
        "城镇(urban)": "1",  # urban -> U_R=1
        "乡村(rural)": "2",  # rural -> U_R=2
    }

    for sheet_name, u_r in sheet_mapping.items():
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols=URBAN_CITY_DISPOSABLE_REQUIRED_COLS,
        )

        # Build lookup for each year found in the sheet (dynamic years)
        available_years = sorted(df["year"].unique().tolist())

        for year in available_years:
            key = f"{u_r}_{year}"
            year_df = df[df["year"] == year].copy()

            # Build lookup; for direct-administered municipalities use 2-digit province code as key
            lookup = {}
            for _, row in year_df.iterrows():
                city_code = _normalize_geo_code_value(row["city_code"])
                if city_code is None:
                    continue
                income = row["可支配收入"]
                province_code = city_code[:2]
                # Direct-administered municipality: use 2-digit province code as key
                if province_code in DIRECT_MUNICIPALITY_CODES:
                    city_key = province_code
                else:
                    city_key = city_code
                lookup[city_key] = float(income)

            result[key] = lookup

    return result


def load_urban_industry_yearly_wage_lookup() -> Tuple[Dict[Tuple[str, int, str], float], Dict[Tuple[str, int, str], float]]:
    path = PathConfig.URBAN_PROVINCIAL_INDUSTRY_WAGE
    df = pd.read_excel(path)

    lookup_by_year: Dict[int, Dict[Tuple[str, int, str], float]] = {2018: {}, 2020: {}}
    required_cols = URBAN_WAGE_REQUIRED_COLS + NONPRIVATE_WAGE_COLS + PRIVATE_WAGE_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Wage table missing required columns: {missing_cols}")

    for _, row in df.iterrows():
        year = int(row["年份"])
        if year not in lookup_by_year:
            continue
        province_code = _normalize_geo_code_value(row["行政区划代码"])
        if province_code is None:
            raise ValueError("Wage table contains empty administrative region codes")
        province_code = province_code[:2]

        for industry_code, col_name in enumerate(NONPRIVATE_WAGE_COLS, start=1):
            value = row[col_name]
            if pd.isna(value):
                continue
            lookup_by_year[year][(province_code, industry_code, "1")] = float(value)

        for industry_code, col_name in enumerate(PRIVATE_WAGE_COLS, start=1):
            value = row[col_name]
            if pd.isna(value):
                if industry_code != 19:
                    continue
                value = lookup_by_year[year].get((province_code, industry_code, "1"))
                if value is None:
                    continue
            lookup_by_year[year][(province_code, industry_code, "2")] = float(value)

    return lookup_by_year[2018], lookup_by_year[2020]


# ============================================================================
# External Validation Data
# ============================================================================

def load_external_city_wage(filepath: Path) -> pd.DataFrame:
    """
    Load external 2020 city-level wage data.
    Returns DataFrame with columns: city_code, avg_wage_annual
    """
    df = pd.read_excel(filepath, sheet_name="Sheet1")
    df = df[["行政区划代码", "职工平均工资(元)"]].copy()
    df.columns = ["city_code", "avg_wage_annual"]
    df["city_code"] = _normalize_geo_code_series(df["city_code"])
    return df


def load_external_rural_income_data(
    province_whitelist: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load external county-level rural disposable income data.
    Returns DataFrame with columns: county_code, rural_income, province_code, region
    """
    income_path = PathConfig.VALIDATION_RURAL_COUNTY_INCOME
    df = pd.read_excel(income_path)

    df["county_code"] = _normalize_geo_code_series(df["区县代码"])
    df["rural_income"] = df["农村居民人均可支配收入_元"]

    df = df[df["county_code"].notna() & (df["county_code"] != "") & df["rural_income"].notna()].copy()

    df["province_code"] = df["county_code"].str[:2]
    df["region"] = df["province_code"].apply(get_region)

    if province_whitelist is not None:
        df = df[df["province_code"].isin(province_whitelist)].copy()

    n_unique = df["county_code"].nunique()
    if n_unique < len(df):
        df = df.drop_duplicates(subset=["county_code"], keep="first")

    return df[["county_code", "rural_income", "province_code", "region"]]


def load_total_pop_lookup(
    province_whitelist: set,
) -> Dict[Tuple[str, str, str], float]:
    """
    Load total population lookup by (province_code, county_code, U_R).
    """
    df = pd.read_parquet(PathConfig.MARGIN_TOTAL_POP, columns=["city_code", "county_code", "U_R", "total_pop"])
    df["city_code"] = df["city_code"].astype("string")
    df["county_code"] = df["county_code"].astype("string")
    df["U_R"] = df["U_R"].astype("string")
    df["province_code"] = df["city_code"].str[:2]

    df = df[df["province_code"].isin(province_whitelist) & df["U_R"].isin(["1", "2"])].copy()

    agg = df.groupby(["province_code", "county_code", "U_R"], as_index=False)["total_pop"].sum()
    lookup = {
        (str(r.province_code), str(r.county_code), str(r.U_R)): float(r.total_pop)
        for r in agg.itertuples(index=False)
    }
    return lookup
