from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

IPF_WEIGHTS_DIR = REPO_ROOT / "output" / "ipf" / "weights"
MICRO_DATA_PATH = REPO_ROOT / "input_data" / "chip_employed.parquet"
MACRO_MASTER_PATH = REPO_ROOT / "input_data" / "macro_master.parquet"

QISI_OUTPUT_DIR = REPO_ROOT / "output" / "integerisation" / "qisi_output"
PREMERGED_OUTPUT_DIR = REPO_ROOT / "output" / "integerisation" / "premerged"

ID_COLUMN = "ID"
WEIGHT_COLUMN = "weight"
INTEGER_WEIGHT_COLUMN = "integer_weight"
STRATUM_ID_COLUMN = "stratum_id"
GEO_CODE_COLUMN = "geo_code"
U_R_COLUMN = "U_R"
SURVEY_YEAR = "2018"

MACRO_VARIABLE_COLUMN = "variable"
MACRO_COUNT_COLUMN = "count"
HARD_CONSTRAINT_VARIABLE = "C_INDUSTRY"

STRATIFY_PRIORITY = [
    "C_INDUSTRY",
    "C_OCCUPATION",
    "C_EDU_WORKER",
    "AGE_BAND",
    "C_SEX",
]

OUTPUT_COLUMNS = [
    "ID",
    "stratum_id",
    "geo_code",
    "U_R",
    "weight",
    "integer_weight",
    "C_INDUSTRY",
    "C_OCCUPATION",
    "C_EDU_WORKER",
    "AGE_BAND",
    "C_SEX",
    "coun",
    "hybrid_annual_wage",
    "company_ownership",
    "full_no_full",
]

QISI_SEED = 123
ENABLE_PARALLEL = True
N_WORKERS = 16
BATCH_SIZE = 50
