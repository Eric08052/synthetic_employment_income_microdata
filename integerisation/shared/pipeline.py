from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from .data_loader import (
    get_target_count,
    get_weight_file_stratum_ids,
    load_ipf_weights,
    load_macro_master,
    load_micro_data,
    merge_weights_with_micro,
)
from .province_outputs import build_premerged_outputs
from .qisi_core import perform_qisi
from .utils import ensure_directory, parse_cc_from_stratum_id, stratum_id_to_filename


def _get_geo_code(stratum_id: str) -> str:
    return str(stratum_id).split("|", 1)[0].strip()


def _get_required_micro_columns(micro_df: pd.DataFrame) -> list[str]:
    available_columns = list(micro_df.columns)
    missing = [column for column in config.STRATIFY_PRIORITY if column not in available_columns]
    if missing:
        raise ValueError(f"Missing required stratification columns in microdata: {missing}")

    required_columns = list(config.STRATIFY_PRIORITY)
    skip_columns = {
        config.ID_COLUMN,
        config.STRATUM_ID_COLUMN,
        config.GEO_CODE_COLUMN,
        config.U_R_COLUMN,
        config.WEIGHT_COLUMN,
        config.INTEGER_WEIGHT_COLUMN,
    }
    for column in config.OUTPUT_COLUMNS:
        if column in skip_columns or column in required_columns:
            continue
        if column in available_columns:
            required_columns.append(column)
    return required_columns


def _align_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for column in config.OUTPUT_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[config.OUTPUT_COLUMNS]


def process_single_stratum(
    stratum_id: str,
    micro_df: pd.DataFrame,
    macro_master: pd.DataFrame,
    output_dir: Path,
    weights_dir: Path,
) -> bool:
    try:
        weights_df = load_ipf_weights(stratum_id, weights_dir=weights_dir)
        required_micro_cols = _get_required_micro_columns(micro_df)
        merged_df = merge_weights_with_micro(weights_df, micro_df, required_micro_cols)
        merged_df[config.STRATUM_ID_COLUMN] = stratum_id
        merged_df[config.GEO_CODE_COLUMN] = _get_geo_code(stratum_id)
        merged_df[config.U_R_COLUMN] = parse_cc_from_stratum_id(stratum_id)

        qisi_df = perform_qisi(
            df=merged_df,
            target_count=get_target_count(stratum_id, macro_master),
            stratify_cols=config.STRATIFY_PRIORITY,
            weight_col=config.WEIGHT_COLUMN,
            integer_weight_col=config.INTEGER_WEIGHT_COLUMN,
            seed=config.QISI_SEED,
        )
        output_path = output_dir / stratum_id_to_filename(stratum_id)
        _align_output_columns(qisi_df).to_parquet(output_path, index=False, compression="snappy")
        return True
    except Exception as exc:
        return False


_worker_micro_df: Optional[pd.DataFrame] = None
_worker_macro_master: Optional[pd.DataFrame] = None


def _worker_init(micro_data_path: Optional[Path] = None, macro_master_path: Optional[Path] = None) -> None:
    global _worker_micro_df, _worker_macro_master
    _worker_micro_df = load_micro_data(micro_data_path)
    _worker_macro_master = load_macro_master(macro_master_path)


def _process_stratum_parallel(args: tuple[str, Path]) -> bool:
    stratum_id, output_dir, weights_dir = args
    return process_single_stratum(
        stratum_id=stratum_id,
        micro_df=_worker_micro_df,
        macro_master=_worker_macro_master,
        output_dir=output_dir,
        weights_dir=weights_dir,
    )


def _get_n_workers(n_strata: int) -> int:
    if not config.ENABLE_PARALLEL or n_strata <= 1:
        return 1
    return max(1, min(config.N_WORKERS, n_strata))


def run_qisi_pipeline(
    micro_data_path: Optional[Path] = None,
    weights_dir: Optional[Path] = None,
    macro_master_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, object]:
    start_time = time.time()

    if output_dir is None:
        out_dir = config.QISI_OUTPUT_DIR
        premerged_dir = config.PREMERGED_OUTPUT_DIR
    else:
        output_root = Path(output_dir)
        out_dir = output_root / "qisi_output"
        premerged_dir = output_root / "premerged"
    ensure_directory(out_dir)
    ensure_directory(premerged_dir)
    source_weights_dir = weights_dir or config.IPF_WEIGHTS_DIR

    stratum_ids = get_weight_file_stratum_ids(source_weights_dir)
    micro_df = load_micro_data(micro_data_path)
    macro_master = load_macro_master(macro_master_path)

    n_workers = _get_n_workers(len(stratum_ids))
    success_count = 0
    fail_count = 0
    successful_stratum_ids: list[str] = []

    if n_workers > 1:
        tasks = [(stratum_id, out_dir, source_weights_dir) for stratum_id in stratum_ids]
        for batch_start in range(0, len(tasks), config.BATCH_SIZE):
            batch_tasks = tasks[batch_start:batch_start + config.BATCH_SIZE]
            with mp.Pool(
                processes=n_workers,
                initializer=_worker_init,
                initargs=(micro_data_path, macro_master_path),
            ) as pool:
                results = pool.map(_process_stratum_parallel, batch_tasks)
            successful_stratum_ids.extend(
                stratum_id
                for (stratum_id, _, _), result in zip(batch_tasks, results)
                if result
            )
            success_count += sum(1 for result in results if result)
            fail_count += sum(1 for result in results if not result)
    else:
        for stratum_id in stratum_ids:
            result = process_single_stratum(
                stratum_id=stratum_id,
                micro_df=micro_df,
                macro_master=macro_master,
                output_dir=out_dir,
                weights_dir=source_weights_dir,
            )
            if result:
                success_count += 1
                successful_stratum_ids.append(stratum_id)
            else:
                fail_count += 1

    premerged_stats = build_premerged_outputs(
        qisi_output_dir=out_dir,
        premerged_dir=premerged_dir,
        output_columns=config.OUTPUT_COLUMNS,
        successful_stratum_ids=successful_stratum_ids,
    )

    return {
        "n_strata": len(stratum_ids),
        "success_count": success_count,
        "fail_count": fail_count,
        "premerged_provinces": len(premerged_stats),
        "premerged_output_dir": str(premerged_dir),
        "duration_seconds": time.time() - start_time,
    }
