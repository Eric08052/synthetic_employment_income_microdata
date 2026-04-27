from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import SYNTHETIC_REQUIRED_COLUMNS
from prepare_common import prepare_source_frame


def prepare_synthetic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_source_frame(
        frame,
        dataset_name="synthetic",
        income_column="syn_income",
        weight_column="integer_weight",
        scope_column="U_R",
        scope_map={"1": "urban", "2": "rural"},
        ownership_column="company_ownership",
        education_column="C_EDU_WORKER",
        occupation_column="C_OCCUPATION",
        ownership_values={"1", "2"},
    )


def load_synthetic_dataset(adjusted_dir: Path) -> pd.DataFrame:
    adjusted_dir = Path(adjusted_dir)
    parquet_files = sorted(adjusted_dir.glob("adjusted_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No adjusted_*.parquet files found in synthetic dir: {adjusted_dir}")

    parts = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=SYNTHETIC_REQUIRED_COLUMNS)
        prepared = prepare_synthetic_frame(frame)
        if not prepared.empty:
            parts.append(prepared)

    if not parts:
        raise ValueError("No usable samples in synthetic data after filtering.")
    return pd.concat(parts, ignore_index=True)
