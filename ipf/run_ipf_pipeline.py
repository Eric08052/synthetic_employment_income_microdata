from __future__ import annotations

import argparse
from pathlib import Path

from shared.pipeline import run_ipf_pipeline_with_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IPF pipeline with optional path overrides.")
    parser.add_argument("--microdata-path", type=Path, default=None)
    parser.add_argument("--task-list-path", type=Path, default=None)
    parser.add_argument("--macro-master-path", type=Path, default=None)
    parser.add_argument("--margin-total-pop-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ipf_pipeline_with_overrides(
        microdata_path=args.microdata_path,
        task_list_path=args.task_list_path,
        macro_master_path=args.macro_master_path,
        margin_total_pop_path=args.margin_total_pop_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
