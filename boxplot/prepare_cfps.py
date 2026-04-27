from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CFPS_REQUIRED_COLUMNS
from prepare_common import prepare_source_frame


def prepare_cfps_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_source_frame(
        frame,
        dataset_name="cfps2020",
        income_column="employment_income",
        weight_column="weight",
        scope_column="urban_rural",
        scope_map={"城镇": "urban", "农村": "rural"},
        ownership_column="ownership_code",
        education_column="education_7_code",
        occupation_column="occupation_7_code",
        ownership_values={"1", "2"},
    )


def load_cfps_dataset(cfps_path: Path) -> pd.DataFrame:
    cfps_path = Path(cfps_path)
    if not cfps_path.exists():
        raise FileNotFoundError(f"CFPS2020 parquet not found: {cfps_path}")
    frame = pd.read_parquet(cfps_path, columns=CFPS_REQUIRED_COLUMNS)
    return prepare_cfps_frame(frame)
