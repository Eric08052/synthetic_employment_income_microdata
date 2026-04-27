"""
Income calibration configuration for the fixed 2020 workflow.
"""

from pathlib import Path
from typing import Dict, List, Set

REGION_GROUPS: Dict[str, tuple[str, ...]] = {
    "East": ("11", "12", "13", "21", "31", "32", "33", "35", "37", "44", "46"),
    "Central": ("14", "22", "23", "34", "36", "41", "42", "43"),
    "West": ("15", "45", "50", "51", "52", "53", "54", "61", "62", "63", "64", "65"),
}

REGION_CLASSIFICATION: Dict[str, str] = {
    province_code: region_name
    for region_name, province_codes in REGION_GROUPS.items()
    for province_code in province_codes
}

DIRECT_MUNICIPALITY_CODES: Set[str] = {"11", "12", "31", "50"}

SURVEY_YEAR = 2018


class PathConfig:
    """Fixed path configuration for the 2020 workflow."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DIR = PROJECT_ROOT / "input_data"
    PREMERGED_DIR = PROJECT_ROOT / "output" / "integerisation" / "premerged"
    GEO_MAPPING_JSON = BASE_DIR / "geo_code_mapping_2020.json"
    URBAN_CITY_DISPOSABLE_INCOME = BASE_DIR / "Per_capita_disposable_income_by_city_2018–2020.xlsx"
    URBAN_PROVINCIAL_INDUSTRY_WAGE = BASE_DIR / "Provincia_wage_data_for_private_and_non-private_sector_employment_2018–2020.xlsx"
    VALIDATION_URBAN_CITY_WAGE = BASE_DIR / "Employment_and_wage_data_by_city_2020.xlsx"
    VALIDATION_RURAL_COUNTY_INCOME = BASE_DIR / "Per_capita_disposable_income_by_district_2020.xlsx"
    MARGIN_TOTAL_POP = BASE_DIR / "margin_total_pop.parquet"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "income_adj"
    ADJUSTED_DIR = OUTPUT_DIR / "adjusted"
    FINAL_OUTPUT_DIR = OUTPUT_DIR / "final_output_data"
    VALIDATION_URBAN_OUTPUT_DIR = OUTPUT_DIR / "validation" / "urban"
    VALIDATION_URBAN_BASELINE_OUTPUT_DIR = VALIDATION_URBAN_OUTPUT_DIR / "baseline"
    VALIDATION_URBAN_ADJUSTED_OUTPUT_DIR = VALIDATION_URBAN_OUTPUT_DIR / "adjusted"
    VALIDATION_RURAL_OUTPUT_DIR = OUTPUT_DIR / "validation" / "rural"
    VALIDATION_RURAL_BASELINE_OUTPUT_DIR = VALIDATION_RURAL_OUTPUT_DIR / "baseline"
    VALIDATION_RURAL_ADJUSTED_OUTPUT_DIR = VALIDATION_RURAL_OUTPUT_DIR / "adjusted"

    @classmethod
    def get_premerged_path(cls, province_code: str) -> Path:
        return cls.PREMERGED_DIR / f"province_{province_code}.parquet"

    @classmethod
    def list_premerged_provinces(cls) -> List[str]:
        if not cls.PREMERGED_DIR.exists():
            return []
        province_files = cls.PREMERGED_DIR.glob("province_*.parquet")
        return sorted(pf.stem.replace("province_", "") for pf in province_files)

    @classmethod
    def get_adjusted_dir(cls) -> Path:
        return cls.ADJUSTED_DIR

    @classmethod
    def get_final_output_dir(cls) -> Path:
        return cls.FINAL_OUTPUT_DIR


class ProvinceConfig:
    """Province metadata for the 2020 workflow."""

    PROVINCE_NAME_TO_CODE: Dict[str, str] = {
        "北京市": "11", "天津市": "12", "河北省": "13", "山西省": "14",
        "内蒙古自治区": "15", "辽宁省": "21", "吉林省": "22", "黑龙江省": "23",
        "上海市": "31", "江苏省": "32", "浙江省": "33", "安徽省": "34",
        "福建省": "35", "江西省": "36", "山东省": "37", "河南省": "41",
        "湖北省": "42", "湖南省": "43", "广东省": "44", "广西壮族自治区": "45",
        "海南省": "46", "重庆市": "50", "四川省": "51", "贵州省": "52",
        "云南省": "53", "西藏自治区": "54", "陕西省": "61", "甘肃省": "62",
        "青海省": "63", "宁夏回族自治区": "64", "新疆维吾尔自治区": "65",
    }
    PROVINCE_CODE_TO_NAME: Dict[str, str] = {v: k for k, v in PROVINCE_NAME_TO_CODE.items()}

    CITY_GRANULARITY_PROVINCES: Set[str] = {"13", "14", "21", "22", "41", "43", "45", "51"}
    PROVINCE_GRANULARITY: Dict[str, str] = {
        province_code: "city"
        for province_code in CITY_GRANULARITY_PROVINCES
    }
    NO_TRACK_PROVINCE_CODES: Set[str] = {"65"}

    GEO_CODE_CORRECTIONS: Dict[str, str] = {
        "340891": "340811",
        "320899": "320804",
    }
    SPECIAL_CITY_SCALE_MAPPING: Dict[str, str] = {
        "419001": "411700",
    }

    @classmethod
    def map_city_scale_geo_code(cls, geo_code: str) -> str:
        return cls.SPECIAL_CITY_SCALE_MAPPING.get(str(geo_code), str(geo_code))

    @classmethod
    def get_geo_granularity(cls, province_code: str) -> str:
        return cls.PROVINCE_GRANULARITY.get(str(province_code), "district")

    @classmethod
    def is_city_granularity(cls, province_code: str) -> bool:
        return cls.get_geo_granularity(province_code) == "city"
