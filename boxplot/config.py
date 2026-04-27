from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = REPO_ROOT / "output" / "income_adj" / "adjusted"
OUTPUT_DIR = REPO_ROOT / "output" / "boxplot"
CFPS_PATH = REPO_ROOT / "input_data" / "cfps2020_employed_micro.parquet"
CHIP_PATH = REPO_ROOT / "input_data" / "original_chip2018.parquet"
SCOPE_ORDER = ["urban", "rural"]
VARIABLE_DATASETS = {
    "education": ["synthetic", "chip2018", "cfps2020"],
    "occupation": ["synthetic", "chip2018", "cfps2020"],
}
VARIABLE_OUTPUT_STEMS = {
    "education": "by_education",
    "occupation": "by_occupation",
}
DATASET_LABELS = {"synthetic": "syn", "chip2018": "CHIP2018", "cfps2020": "CFPS2020"}
DATASET_COLORS = {"synthetic": "#FC9B18", "chip2018": "#43C5D2", "cfps2020": "#C4B0D9"}
DATASET_PLOT_ORDER = ["synthetic", "chip2018", "cfps2020"]
VARIABLE_EXCLUDED_CODES = {"occupation": {"7"}}
EDUCATION_CATEGORY_CODES = {str(code) for code in range(1, 8)}
OCCUPATION_CATEGORY_CODES = {str(code) for code in range(1, 8)}

CFPS_REQUIRED_COLUMNS = [
    "employment_income",
    "weight",
    "urban_rural",
    "occupation_7_code",
    "education_7_code",
    "ownership_code",
]
SYNTHETIC_REQUIRED_COLUMNS = [
    "syn_income",
    "integer_weight",
    "company_ownership",
    "C_OCCUPATION",
    "C_EDU_WORKER",
    "U_R",
]
CHIP_REQUIRED_COLUMNS = [
    "hybrid_annual_wage",
    "company_ownership",
    "C_OCCUPATION",
    "C_EDU_WORKER",
    "U_R",
]
