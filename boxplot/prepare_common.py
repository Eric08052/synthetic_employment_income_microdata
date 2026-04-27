from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import EDUCATION_CATEGORY_CODES, OCCUPATION_CATEGORY_CODES
from utils import normalize_code_series, require_columns


OUTPUT_COLUMNS = [
    "dataset_name",
    "scope_name",
    "variable_name",
    "category_code",
    "category_label",
    "income",
    "weight",
    "sample_n",
]


def _build_variable_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    variable_name: str,
    code_column: str,
    valid_codes: set[str],
) -> pd.DataFrame:
    working = frame[[code_column, "scope_name", "income", "weight"]].copy()
    working["category_code"] = normalize_code_series(working[code_column])
    working = working.loc[working["category_code"].isin(valid_codes)].copy()
    working["category_label"] = working["category_code"]
    working["dataset_name"] = dataset_name
    working["variable_name"] = variable_name
    working["sample_n"] = 1
    return working.loc[:, OUTPUT_COLUMNS]


def prepare_source_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    income_column: str,
    weight_column: str | None,
    scope_column: str,
    scope_map: dict[str, str],
    ownership_column: str,
    education_column: str,
    occupation_column: str,
    ownership_values: set[str],
) -> pd.DataFrame:
    required_columns = [
        income_column,
        ownership_column,
        education_column,
        occupation_column,
        scope_column,
    ]
    if weight_column is not None:
        required_columns.append(weight_column)
    require_columns(frame, required_columns, dataset_name)

    working = frame.copy()
    working[ownership_column] = normalize_code_series(working[ownership_column])
    working[scope_column] = normalize_code_series(working[scope_column])
    working["scope_name"] = working[scope_column].map(scope_map)
    working["income"] = pd.to_numeric(working[income_column], errors="coerce")
    if weight_column is None:
        working["weight"] = 1.0
    else:
        working["weight"] = pd.to_numeric(working[weight_column], errors="coerce")

    mask = working[ownership_column].isin(ownership_values)
    mask &= working[scope_column].isin(scope_map)
    mask &= working["income"].notna() & working["income"].gt(0)
    mask &= working["weight"].notna() & working["weight"].gt(0)
    mask &= working["scope_name"].notna()
    scoped = working.loc[mask].copy()

    parts = [
        _build_variable_frame(
            scoped,
            dataset_name=dataset_name,
            variable_name="education",
            code_column=education_column,
            valid_codes=EDUCATION_CATEGORY_CODES,
        ),
        _build_variable_frame(
            scoped,
            dataset_name=dataset_name,
            variable_name="occupation",
            code_column=occupation_column,
            valid_codes=OCCUPATION_CATEGORY_CODES,
        ),
    ]
    return pd.concat(parts, ignore_index=True)


def load_source_dataset(
    data_path: Path,
    *,
    dataset_name: str,
    columns: list[str],
    income_column: str,
    weight_column: str | None,
    scope_column: str,
    scope_map: dict[str, str],
    ownership_column: str,
    education_column: str,
    occupation_column: str,
    ownership_values: set[str],
) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{dataset_name} parquet not found: {data_path}")
    frame = pd.read_parquet(data_path, columns=columns)
    return prepare_source_frame(
        frame,
        dataset_name=dataset_name,
        income_column=income_column,
        weight_column=weight_column,
        scope_column=scope_column,
        scope_map=scope_map,
        ownership_column=ownership_column,
        education_column=education_column,
        occupation_column=occupation_column,
        ownership_values=ownership_values,
    )
