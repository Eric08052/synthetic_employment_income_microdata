from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STAGE_ORDER = (
    "ipf",
    "integerisation",
    "income_adj",
    "validation",
    "boxplot",
)

SUBPROCESS_ENV_OVERRIDES = {
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp/mplconfig",
    "PYTHONFAULTHANDLER": "1",
}


def build_output_layout(output_root: Path) -> dict[str, Path]:
    ipf_dir = output_root / "ipf"
    integerisation_dir = output_root / "integerisation"
    income_adj_dir = output_root / "income_adj"
    boxplot_dir = output_root / "boxplot"
    return {
        "output_root": output_root,
        "ipf_dir": ipf_dir,
        "ipf_weights_dir": ipf_dir / "weights",
        "integerisation_dir": integerisation_dir,
        "integerisation_qisi_dir": integerisation_dir / "qisi_output",
        "premerged_dir": integerisation_dir / "premerged",
        "income_adj_dir": income_adj_dir,
        "income_adj_adjusted_dir": income_adj_dir / "adjusted",
        "income_adj_final_output_dir": income_adj_dir / "final_output_data",
        "boxplot_dir": boxplot_dir,
    }


def build_stage_commands(
    project_root: Path,
    python_executable: Path,
    micro_path: Path,
    layout: dict[str, Path],
    selected_stages: set[str] | None = None,
) -> list[dict[str, object]]:
    input_dir = project_root / "input_data"
    commands = [
        {
            "name": "ipf",
            "stage": "ipf",
            "argv": [
                str(python_executable),
                str(project_root / "ipf" / "run_ipf_pipeline.py"),
                "--microdata-path",
                str(micro_path),
                "--task-list-path",
                str(input_dir / "task_list.parquet"),
                "--macro-master-path",
                str(input_dir / "macro_master.parquet"),
                "--margin-total-pop-path",
                str(input_dir / "margin_total_pop.parquet"),
                "--output-dir",
                str(layout["ipf_dir"]),
            ],
        },
        {
            "name": "integerisation",
            "stage": "integerisation",
            "argv": [
                str(python_executable),
                str(project_root / "integerisation" / "run_qisi_main.py"),
                "--micro-data-path",
                str(micro_path),
                "--weights-dir",
                str(layout["ipf_weights_dir"]),
                "--macro-master-path",
                str(input_dir / "macro_master.parquet"),
                "--output-dir",
                str(layout["integerisation_dir"]),
            ],
        },
        {
            "name": "income_adj",
            "stage": "income_adj",
            "argv": [
                str(python_executable),
                str(project_root / "income_adj" / "main.py"),
                "--premerged-dir",
                str(layout["premerged_dir"]),
                "--output-dir",
                str(layout["income_adj_dir"]),
            ],
        },
        {
            "name": "validation_urban",
            "stage": "validation",
            "argv": [
                str(python_executable),
                str(project_root / "income_adj" / "validation_urban.py"),
                "--premerged-dir",
                str(layout["premerged_dir"]),
                "--output-dir",
                str(layout["income_adj_dir"]),
            ],
        },
        {
            "name": "validation_rural",
            "stage": "validation",
            "argv": [
                str(python_executable),
                str(project_root / "income_adj" / "validation_rural.py"),
                "--premerged-dir",
                str(layout["premerged_dir"]),
                "--output-dir",
                str(layout["income_adj_dir"]),
            ],
        },
        {
            "name": "boxplot",
            "stage": "boxplot",
            "argv": [
                str(python_executable),
                str(project_root / "boxplot" / "violin_panel.py"),
                "--synthetic-dir",
                str(layout["income_adj_adjusted_dir"]),
                "--chip-path",
                str(micro_path),
                "--output-dir",
                str(layout["boxplot_dir"]),
            ],
        },
    ]
    if selected_stages is None:
        return commands
    return [command for command in commands if command["stage"] in selected_stages]


def ensure_output_dirs(layout: dict[str, Path]) -> None:
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)


def build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(SUBPROCESS_ENV_OVERRIDES)
    return env


def run_command(argv: list[str], cwd: Path) -> tuple[int, float]:
    started_at = time.time()
    completed = subprocess.run(argv, cwd=cwd, env=build_subprocess_env(), check=False)
    return completed.returncode, time.time() - started_at


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_stage_selection(raw_value: str | None) -> set[str] | None:
    if raw_value is None:
        return None
    values = {item.strip() for item in raw_value.split(",") if item.strip()}
    invalid_values = values - set(STAGE_ORDER)
    if invalid_values:
        invalid_text = ", ".join(sorted(invalid_values))
        raise ValueError(f"Invalid stage names: {invalid_text}")
    return values


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the full single-version workflow under output/.")
    parser.add_argument("--micro-path", type=Path, default=project_root / "input_data" / "chip_employed.parquet")
    parser.add_argument("--output-root", type=Path, default=project_root / "output")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage groups: ipf,integerisation,income_adj,validation,boxplot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    selected_stages = parse_stage_selection(args.stages)

    micro_path = Path(args.micro_path)
    if not micro_path.exists():
        raise FileNotFoundError(f"Microdata parquet not found: {micro_path}")

    output_root = Path(args.output_root)
    layout = build_output_layout(output_root)
    ensure_output_dirs(layout)

    commands = build_stage_commands(
        project_root=project_root,
        python_executable=Path(args.python_executable),
        micro_path=micro_path,
        layout=layout,
        selected_stages=selected_stages,
    )

    summary = {
        "micro_path": str(micro_path),
        "output_root": str(output_root),
        "status": "success",
        "stages": [],
    }
    total_started_at = time.time()
    for command in commands:
        returncode, duration_seconds = run_command(command["argv"], cwd=project_root)
        summary["stages"].append(
            {
                "name": command["name"],
                "returncode": returncode,
                "duration_seconds": round(duration_seconds, 3),
            }
        )
        if returncode != 0:
            summary["status"] = "failed"
            summary["failed_stage"] = command["name"]
            break
    summary["duration_seconds"] = round(time.time() - total_started_at, 3)
    write_json(output_root / "workflow_summary.json", summary)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
