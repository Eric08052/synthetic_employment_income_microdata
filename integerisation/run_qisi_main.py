from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from shared.pipeline import run_qisi_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integerisation pipeline with optional path overrides.")
    parser.add_argument("--micro-data-path", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--macro-master-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_qisi_pipeline(
        micro_data_path=args.micro_data_path,
        weights_dir=args.weights_dir,
        macro_master_path=args.macro_master_path,
        output_dir=args.output_dir,
    )
    print(summary)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
