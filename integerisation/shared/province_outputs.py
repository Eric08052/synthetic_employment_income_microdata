from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Collection, Dict, List, Optional

import pandas as pd

from .utils import filename_to_stratum_id, stratum_id_to_filename, write_parquet


def extract_province_code_from_stratum_id(stratum_id: str) -> str:
    return str(stratum_id).split("|", 1)[0].strip()[:2]


def _optimize_premerged_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()

    if "integer_weight" in optimized.columns:
        optimized["integer_weight"] = pd.to_numeric(
            optimized["integer_weight"],
            errors="coerce",
        ).astype("Int32")
    if "U_R" in optimized.columns:
        optimized["U_R"] = optimized["U_R"].astype("string")
    if "coun" in optimized.columns:
        optimized["coun"] = optimized["coun"].astype("string")
    if "stratum_id" in optimized.columns:
        optimized["stratum_id"] = optimized["stratum_id"].astype("string")
    if "hybrid_annual_wage" in optimized.columns:
        optimized["hybrid_annual_wage"] = pd.to_numeric(
            optimized["hybrid_annual_wage"],
            errors="coerce",
        ).astype("float32")
    if "C_INDUSTRY" in optimized.columns:
        optimized["C_INDUSTRY"] = pd.to_numeric(
            optimized["C_INDUSTRY"],
            errors="coerce",
        ).astype("Int16")
    if "company_ownership" in optimized.columns:
        optimized["company_ownership"] = optimized["company_ownership"].astype("string")
    if "full_no_full" in optimized.columns:
        optimized["full_no_full"] = optimized["full_no_full"].astype("string")
    return optimized


def build_premerged_outputs(
    qisi_output_dir: Path,
    premerged_dir: Path,
    output_columns: List[str],
    successful_stratum_ids: Optional[Collection[str]] = None,
) -> Dict[str, Dict[str, object]]:
    premerged_dir.mkdir(parents=True, exist_ok=True)

    allowed_filenames = None
    if successful_stratum_ids is not None:
        allowed_filenames = {
            stratum_id_to_filename(stratum_id)
            for stratum_id in successful_stratum_ids
        }

    province_files: Dict[str, List[Path]] = defaultdict(list)
    for parquet_file in sorted(qisi_output_dir.glob("*.parquet")):
        if allowed_filenames is not None and parquet_file.name not in allowed_filenames:
            continue
        stratum_id = filename_to_stratum_id(parquet_file.name)
        province_code = extract_province_code_from_stratum_id(stratum_id)
        if province_code:
            province_files[province_code].append(parquet_file)

    stats: Dict[str, Dict[str, object]] = {}
    for province_code, file_paths in sorted(province_files.items()):
        frames = [pd.read_parquet(file_path, columns=output_columns) for file_path in file_paths]
        combined = pd.concat(frames, ignore_index=True)
        combined = _optimize_premerged_dtypes(combined[output_columns])

        output_path = premerged_dir / f"province_{province_code}.parquet"
        write_parquet(combined, output_path)
        stats[province_code] = {
            "n_files": len(file_paths),
            "n_rows": int(len(combined)),
            "output_path": str(output_path),
        }

    return stats
