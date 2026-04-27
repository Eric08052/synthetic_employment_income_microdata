from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config
from .core import build_constraint_matrix, ipf_iterate


@dataclass
class RuntimeResult:
    stratum_id: str
    iterations: int
    max_weight: float


def build_weights_output_df(
    seed_df: pd.DataFrame,
    indices: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    base = seed_df.loc[indices, [config.MICRO_ID_COLUMN]].copy()
    base["weight"] = weights
    return base[[config.MICRO_ID_COLUMN, "weight"]]


def _run_ipf_once(
    initial_weights: np.ndarray,
    targets_dict: Dict[str, Dict],
    indicators_dict: Dict[str, Dict],
) -> Tuple[np.ndarray, int]:
    return ipf_iterate(
        weights=initial_weights,
        targets_dict=targets_dict,
        indicators_dict=indicators_dict,
        cap_lower=config.CAP_LOWER,
        max_iter=config.MAX_ITERATIONS,
        tol=config.CONVERGENCE_THRESHOLD,
    )


def run_single_ipf_with_constraints(
    stratum_id: str,
    seed_df: pd.DataFrame,
    constraints: pd.DataFrame,
    target_columns: list[str],
    geo_code: str,
) -> Tuple[Optional[RuntimeResult], Optional[np.ndarray], Optional[np.ndarray]]:
    targets_dict, indicators_dict = build_constraint_matrix(
        seed_df,
        constraints,
        target_columns,
    )
    if not targets_dict:
        return None, None, None

    initial_weights = config.compute_initial_weights(seed_df, geo_code)
    final_weights, iterations = _run_ipf_once(
        initial_weights=initial_weights,
        targets_dict=targets_dict,
        indicators_dict=indicators_dict,
    )
    return (
        RuntimeResult(
            stratum_id=stratum_id,
            iterations=iterations,
            max_weight=float(np.max(final_weights)),
        ),
        final_weights,
        seed_df.index.values,
    )


def run_phase1_single(
    stratum_id: str,
    geo_code: str,
    cc_value: str,
    micro_subsets: Dict[str, pd.DataFrame],
    macro_master: pd.DataFrame,
    zero_total_pop_strata: set[str],
) -> Tuple[Optional[RuntimeResult], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if stratum_id in zero_total_pop_strata:
            return None, None, None

        seed_df = micro_subsets.get(cc_value)
        if seed_df is None or seed_df.empty:
            return None, None, None

        constraints = macro_master[macro_master["stratum_id"] == stratum_id]
        if constraints.empty:
            return None, None, None

        return run_single_ipf_with_constraints(
            stratum_id=stratum_id,
            seed_df=seed_df,
            constraints=constraints,
            target_columns=config.DEFAULT_TARGET_COLUMNS,
            geo_code=geo_code,
        )
    except Exception:
        return None, None, None


_shared_data: Dict[str, object] = {}


def _init_worker(shared_dict: Dict) -> None:
    _shared_data.update(shared_dict)
    weights_dir = shared_dict.get("weights_dir")
    if weights_dir is not None:
        config.WEIGHTS_DIR = Path(weights_dir)


def _save_weights(
    result: RuntimeResult,
    weights: Optional[np.ndarray],
    indices: Optional[np.ndarray],
    micro_subsets: Dict[str, pd.DataFrame],
    cc_value: str,
) -> None:
    if weights is None or indices is None:
        return

    seed_df = micro_subsets.get(cc_value)
    if seed_df is None:
        return

    weights_df = build_weights_output_df(seed_df, indices, weights)
    weights_path = config.WEIGHTS_DIR / f"{config.safe_stratum_id(result.stratum_id)}.parquet"
    weights_df.to_parquet(weights_path, index=False)


def worker_phase1(task_dict: Dict) -> Optional[RuntimeResult]:
    stratum_id = task_dict["stratum_id"]
    geo_code = task_dict["geo_code"]
    cc_value = config.resolve_task_cc(task_dict)

    result, weights, indices = run_phase1_single(
        stratum_id=stratum_id,
        geo_code=geo_code,
        cc_value=cc_value,
        micro_subsets=_shared_data["micro_subsets"],
        macro_master=_shared_data["macro_master"],
        zero_total_pop_strata=_shared_data["zero_total_pop_strata"],
    )
    if result is not None:
        _save_weights(
            result,
            weights,
            indices,
            _shared_data["micro_subsets"],
            cc_value,
        )
    return result


def run_ipf_pipeline() -> Dict[str, object]:
    return run_ipf_pipeline_with_overrides()


def run_ipf_pipeline_with_overrides(
    microdata_path: Optional[Path] = None,
    task_list_path: Optional[Path] = None,
    macro_master_path: Optional[Path] = None,
    margin_total_pop_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    original_output_dir = config.OUTPUT_DIR
    original_weights_dir = config.WEIGHTS_DIR
    original_runtime_path = config.RUNTIME_METADATA_PATH
    original_quality_geo_path = config.QUALITY_BY_GEO_PATH
    original_quality_var_path = config.QUALITY_BY_VARIABLE_CATEGORY_PATH
    original_microdata_path = config.MICRODATA_PATH
    original_task_list_path = config.TASK_LIST_PATH
    original_macro_master_path = config.MACRO_MASTER_PATH
    original_margin_total_pop_path = config.MARGIN_TOTAL_POP_PATH

    effective_output_dir = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    effective_weights_dir = effective_output_dir / "weights"
    effective_runtime_path = effective_output_dir / "ipf_runtime_by_geo.csv"
    effective_quality_geo_path = effective_output_dir / "ipf_results_by_geo.csv"
    effective_quality_var_path = effective_output_dir / "ipf_results_by_variable_category.csv"

    config.OUTPUT_DIR = effective_output_dir
    config.WEIGHTS_DIR = effective_weights_dir
    config.RUNTIME_METADATA_PATH = effective_runtime_path
    config.QUALITY_BY_GEO_PATH = effective_quality_geo_path
    config.QUALITY_BY_VARIABLE_CATEGORY_PATH = effective_quality_var_path
    if microdata_path is not None:
        config.MICRODATA_PATH = Path(microdata_path)
    if task_list_path is not None:
        config.TASK_LIST_PATH = Path(task_list_path)
    if macro_master_path is not None:
        config.MACRO_MASTER_PATH = Path(macro_master_path)
    if margin_total_pop_path is not None:
        config.MARGIN_TOTAL_POP_PATH = Path(margin_total_pop_path)

    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        task_list = pd.read_parquet(config.TASK_LIST_PATH)
        macro_master = pd.read_parquet(config.MACRO_MASTER_PATH)
        total_pop_df = pd.read_parquet(config.MARGIN_TOTAL_POP_PATH)
        microdata = pd.read_parquet(config.MICRODATA_PATH)

        if task_list.empty:
            return {"n_total": 0}

        micro_subsets = config.build_micro_subsets(microdata)
        zero_total_pop_strata = config.get_zero_total_pop_strata(total_pop_df)
        task_dicts = task_list.to_dict("records")

        runtime_results: list[RuntimeResult] = []
        with Pool(
            processes=config.get_worker_count(),
            initializer=_init_worker,
            initargs=(
                dict(
                    micro_subsets=micro_subsets,
                    macro_master=macro_master,
                    zero_total_pop_strata=zero_total_pop_strata,
                    weights_dir=config.WEIGHTS_DIR,
                ),
            ),
        ) as pool:
            for result in pool.imap(worker_phase1, task_dicts):
                if result is not None:
                    runtime_results.append(result)

        runtime_df = pd.DataFrame(
            [
                {
                    "stratum_id": result.stratum_id,
                    "iterations": result.iterations,
                    "max_weight": result.max_weight,
                }
                for result in runtime_results
            ],
            columns=["stratum_id", "iterations", "max_weight"],
        )
        runtime_df = runtime_df.sort_values("stratum_id").reset_index(drop=True)
        runtime_df.to_csv(config.RUNTIME_METADATA_PATH, index=False, encoding="utf-8-sig")
        return {"n_total": len(runtime_results)}
    finally:
        config.OUTPUT_DIR = original_output_dir
        config.WEIGHTS_DIR = original_weights_dir
        config.RUNTIME_METADATA_PATH = original_runtime_path
        config.QUALITY_BY_GEO_PATH = original_quality_geo_path
        config.QUALITY_BY_VARIABLE_CATEGORY_PATH = original_quality_var_path
        config.MICRODATA_PATH = original_microdata_path
        config.TASK_LIST_PATH = original_task_list_path
        config.MACRO_MASTER_PATH = original_macro_master_path
        config.MARGIN_TOTAL_POP_PATH = original_margin_total_pop_path
