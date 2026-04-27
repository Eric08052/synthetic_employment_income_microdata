"""
Income Calibration System - Main Entry

Execute city-level disposable income calibration with memory-efficient streaming.
"""

import gc
import logging
import argparse
from datetime import datetime
from pathlib import Path

from config import PathConfig
from data_loader import (
    load_geo_mapping,
    load_city_disposable_income_triple,
    load_urban_industry_yearly_wage_lookup,
)
from calibration import run_streaming_calibration

logger = logging.getLogger(__name__)


def apply_path_overrides(premerged_dir: Path | None = None, output_dir: Path | None = None) -> None:
    if premerged_dir is not None:
        PathConfig.PREMERGED_DIR = Path(premerged_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        PathConfig.OUTPUT_DIR = output_dir
        PathConfig.ADJUSTED_DIR = output_dir / "adjusted"
        PathConfig.FINAL_OUTPUT_DIR = output_dir / "final_output_data"
        PathConfig.VALIDATION_URBAN_OUTPUT_DIR = output_dir / "validation" / "urban"
        PathConfig.VALIDATION_URBAN_BASELINE_OUTPUT_DIR = PathConfig.VALIDATION_URBAN_OUTPUT_DIR / "baseline"
        PathConfig.VALIDATION_URBAN_ADJUSTED_OUTPUT_DIR = PathConfig.VALIDATION_URBAN_OUTPUT_DIR / "adjusted"
        PathConfig.VALIDATION_RURAL_OUTPUT_DIR = output_dir / "validation" / "rural"
        PathConfig.VALIDATION_RURAL_BASELINE_OUTPUT_DIR = PathConfig.VALIDATION_RURAL_OUTPUT_DIR / "baseline"
        PathConfig.VALIDATION_RURAL_ADJUSTED_OUTPUT_DIR = PathConfig.VALIDATION_RURAL_OUTPUT_DIR / "adjusted"


def setup_logging(output_dir: Path) -> None:
    log_file = output_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.ERROR,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main() -> None:
    output_dir = PathConfig.OUTPUT_DIR
    adjusted_dir = PathConfig.get_adjusted_dir()
    final_output_dir = PathConfig.get_final_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger.info("2020 processing...")

    county_to_city, _, _ = load_geo_mapping()
    rural_city_scale_lookup = load_city_disposable_income_triple()
    urban_origin_lookup, urban_target_lookup = load_urban_industry_yearly_wage_lookup()

    run_streaming_calibration(
        county_to_city=county_to_city,
        rural_city_scale_lookup=rural_city_scale_lookup,
        urban_industry_scale_lookup={
            2018: urban_origin_lookup,
            2020: urban_target_lookup,
        },
        adjusted_dir=adjusted_dir,
        final_output_dir=final_output_dir,
    )

    del rural_city_scale_lookup
    del urban_origin_lookup
    del urban_target_lookup
    gc.collect()

    logger.info("done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run new income calibration with optional path overrides.")
    parser.add_argument("--premerged-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    apply_path_overrides(premerged_dir=cli_args.premerged_dir, output_dir=cli_args.output_dir)
    main()
