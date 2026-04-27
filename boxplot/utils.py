from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_columns(frame: pd.DataFrame, required_columns: list[str], frame_name: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def normalize_code_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
