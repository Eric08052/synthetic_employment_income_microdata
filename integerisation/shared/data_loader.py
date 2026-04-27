from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

import config
from .utils import filename_to_stratum_id, stratum_id_to_filename


def get_weight_file_stratum_ids(weights_dir: Optional[Path] = None) -> List[str]:
    source_dir = weights_dir or config.IPF_WEIGHTS_DIR
    stratum_ids = [
        filename_to_stratum_id(path.name)
        for path in sorted(source_dir.glob("*.parquet"))
        if path.is_file()
    ]
    if not stratum_ids:
        raise FileNotFoundError(f"No weight files found in: {source_dir}")
    return stratum_ids


def load_ipf_weights(stratum_id: str, weights_dir: Optional[Path] = None) -> pd.DataFrame:
    source_dir = weights_dir or config.IPF_WEIGHTS_DIR
    filepath = source_dir / stratum_id_to_filename(stratum_id)
    if not filepath.exists():
        raise FileNotFoundError(f"IPF weight file not found: {filepath}")

    return pd.read_parquet(
        filepath,
        columns=[config.ID_COLUMN, config.WEIGHT_COLUMN],
    )


def load_micro_data(micro_path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_parquet(micro_path or config.MICRO_DATA_PATH)


def load_macro_master(macro_path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_parquet(macro_path or config.MACRO_MASTER_PATH)


def get_target_count(stratum_id: str, macro_master: pd.DataFrame) -> int:
    mask = (
        (macro_master[config.STRATUM_ID_COLUMN] == stratum_id)
        & (macro_master[config.MACRO_VARIABLE_COLUMN] == config.HARD_CONSTRAINT_VARIABLE)
    )
    macro_subset = macro_master[mask]
    if macro_subset.empty:
        raise ValueError(f"Hard constraint variable {config.HARD_CONSTRAINT_VARIABLE} not found for stratum {stratum_id}")
    return int(macro_subset[config.MACRO_COUNT_COLUMN].sum())


def merge_weights_with_micro(
    weights_df: pd.DataFrame,
    micro_df: pd.DataFrame,
    required_cols: List[str],
) -> pd.DataFrame:
    micro_subset = micro_df[[config.ID_COLUMN] + required_cols]
    return weights_df.merge(micro_subset, on=config.ID_COLUMN, how="left")
