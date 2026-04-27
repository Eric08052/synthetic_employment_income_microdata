from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_halton_sequence(n: int, base: int = 2) -> np.ndarray:
    sequence = np.zeros(n)
    for i in range(n):
        factor = 1.0
        value = 0.0
        idx = i + 1
        while idx > 0:
            factor /= base
            value += factor * (idx % base)
            idx //= base
        sequence[i] = value
    return sequence


def scramble_halton_cranley_patterson(halton_seq: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    scrambled = (halton_seq + np.random.uniform(0, 1)) % 1
    scrambled[scrambled == 0] = np.finfo(float).eps
    return scrambled


def stratum_id_to_filename(stratum_id: str) -> str:
    return f"{stratum_id.replace('|', '_')}.parquet"


def filename_to_stratum_id(filename: str) -> str:
    name = Path(filename).name
    if not name.endswith(".parquet"):
        raise ValueError(f"Unsupported weight filename: {filename}")
    stem = Path(name).stem
    head, sep, tail = stem.rpartition("_")
    if sep and tail in {"1", "2"} and head:
        return f"{head}|{tail}"
    return stem


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    df.to_parquet(parquet_path, index=False, compression="snappy")


def parse_cc_from_stratum_id(stratum_id: str) -> int:
    if "|" not in stratum_id:
        raise ValueError(f"Invalid stratum_id: {stratum_id}")
    geo_code, cc_text = stratum_id.split("|", 1)
    if not geo_code or cc_text not in {"1", "2"}:
        raise ValueError(f"Invalid stratum_id: {stratum_id}")
    return int(cc_text)
