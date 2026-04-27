from __future__ import annotations

from pathlib import Path

import multiprocessing
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT.parent / "input_data"
OUTPUT_DIR = PROJECT_ROOT.parent / "output" / "ipf"

MICRODATA_PATH = INPUT_DIR / "micro" / "chip_employed.parquet"
TASK_LIST_PATH = INPUT_DIR / "task_list.parquet"
MACRO_MASTER_PATH = INPUT_DIR / "macro_master.parquet"
MARGIN_TOTAL_POP_PATH = INPUT_DIR / "margin_total_pop.parquet"

WEIGHTS_DIR = OUTPUT_DIR / "weights"
RUNTIME_METADATA_PATH = OUTPUT_DIR / "ipf_runtime_by_geo.csv"
QUALITY_BY_GEO_PATH = OUTPUT_DIR / "ipf_results_by_geo.csv"
QUALITY_BY_VARIABLE_CATEGORY_PATH = OUTPUT_DIR / "ipf_results_by_variable_category.csv"

MICRO_ID_COLUMN = "ID"
MICRO_CC_COLUMN = "U_R"
URBAN_CC = "1"
RURAL_CC = "2"

DEFAULT_TARGET_COLUMNS = ["AGE_BAND", "C_EDU_WORKER", "C_INDUSTRY", "C_OCCUPATION", "C_SEX"]

INITIAL_WEIGHT = 1.0
CAP_LOWER = 0.0
CONVERGENCE_THRESHOLD = 1e-4
MAX_ITERATIONS = 1000

RESERVED_CORES = 15


def get_worker_count() -> int:
    return max(1, multiprocessing.cpu_count() - RESERVED_CORES)


def build_micro_subsets(microdata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subsets: dict[str, pd.DataFrame] = {}
    subsets[URBAN_CC] = microdata[microdata[MICRO_CC_COLUMN] == URBAN_CC].copy()
    subsets[RURAL_CC] = microdata[microdata[MICRO_CC_COLUMN] == RURAL_CC].copy()
    return subsets


def resolve_task_cc(task_dict: dict) -> str:
    cc_raw = task_dict.get("U_R")
    if cc_raw is None:
        raise ValueError(f"stratum {task_dict.get('stratum_id')} missing U_R")
    text = str(cc_raw).strip()
    if text not in {URBAN_CC, RURAL_CC}:
        raise ValueError(f"stratum {task_dict.get('stratum_id')} has invalid U_R: {cc_raw}")
    return text


def compute_initial_weights(seed_df: pd.DataFrame, geo_code: str) -> np.ndarray:
    return np.full(len(seed_df), INITIAL_WEIGHT, dtype=np.float64)


def get_zero_total_pop_strata(total_pop_df: pd.DataFrame) -> set[str]:
    if total_pop_df.empty:
        return set()
    zero_rows = total_pop_df[total_pop_df["total_pop"] == 0]
    return {str(value) for value in zero_rows["stratum_id"].dropna().astype(str).unique()}


def safe_stratum_id(stratum_id: str) -> str:
    return str(stratum_id).replace("|", "_").replace("/", "_").replace("\\", "_")
