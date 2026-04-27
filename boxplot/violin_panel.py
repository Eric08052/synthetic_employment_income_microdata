from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import layout
import prepare_cfps
import prepare_common
import prepare_synthetic
from config import (
    CFPS_PATH,
    CHIP_PATH,
    CHIP_REQUIRED_COLUMNS,
    DATASET_COLORS,
    DATASET_LABELS,
    DATASET_PLOT_ORDER,
    OUTPUT_DIR,
    SCOPE_ORDER,
    SYNTHETIC_DIR,
    VARIABLE_EXCLUDED_CODES,
)
from utils import ensure_directory

DEFAULT_PANEL_VARIABLES = ["education", "occupation"]
EXCLUDED_CATEGORY_CODES = VARIABLE_EXCLUDED_CODES
LOG_OUTPUT_STEM = "log"
Y_AXIS_LABEL = "log(annual employment income)"
VARIABLE_TITLES = {
    "education": "Education",
    "occupation": "Occupation",
}
LEGEND_LABELS = {
    "synthetic": "synthetic data",
    "chip2018": "CHIP2018",
    "cfps2020": "CFPS2020",
}
BOX_FACE_ALPHA = 0.78
BOX_EDGE_COLOR = "#34495E"
GRID_ALPHA = 0.35
VARIABLE_LABEL_FONT_SIZE = 15
X_TICK_LABEL_FONT_SIZE = 12
Y_TICK_LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 14

VIOLIN_FACE_ALPHA = 0.35
VIOLIN_EDGE_ALPHA = 0.85
VIOLIN_EDGE_WIDTH = 0.9
KDE_GRID_POINTS = 128
KDE_BANDWIDTH_SCALE = 1.0
SLOT_WIDTH = layout.GROUP_SLOT_WIDTH
BOX_DRAW_WIDTH = 0.12
VIOLIN_MAX_WIDTH = 0.13
CATEGORY_GAP = 1.72
FIGURE_WIDTH_PER_VARIABLE = 13.2
FIGURE_HEIGHT = 10.8
LAYOUT_RECT = (0.03, 0.14, 0.995, 0.95)
LAYOUT_W_PAD = 5.6
VIOLIN_DISPLAY_LABELS = {
    "education": {
        "1": "No schooling",
        "2": "Primary school",
        "3": "Junior secondary school",
        "4": "Senior secondary school",
        "5": "Junior college",
        "6": "Bachelor's degree",
        "7": "Master's degree and above",
    },
    "occupation": {
        "1": "Senior officials and institutional leaders",
        "2": "Professional and technical personnel",
        "3": "Clerical personnel",
        "4": "Social production and life service personnel",
        "5": "Agricultural personnel",
        "6": "Production and related personnel",
    },
}
VIOLIN_X_LABEL_WRAP_WIDTH = {
    "education": 18,
    "occupation": 18,
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11


def _sortable_category_code(value: object) -> tuple[int, object]:
    if pd.isna(value):
        return (2, "")
    text = str(value).strip()
    if not text:
        return (2, "")
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    sorted_weights = weights[sorter]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative = cumulative / sorted_weights.sum()
    return np.interp(quantiles, cumulative, sorted_values)


def compute_weighted_box_stats(
    frame: pd.DataFrame,
    value_column: str = "plot_value",
    weight_column: str = "weight",
) -> dict[str, float | int | list[float]]:
    valid = frame[[value_column, weight_column]].dropna().copy()
    valid = valid.loc[valid[weight_column].gt(0)].copy()
    if valid.empty:
        raise ValueError("Cannot compute box statistics: no valid samples.")

    values = valid[value_column].astype(float).to_numpy()
    weights = valid[weight_column].astype(float).to_numpy()
    q1, median, q3 = _weighted_quantile(values, weights, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr

    inlier_values = values[(values >= low_fence) & (values <= high_fence)]
    if inlier_values.size == 0:
        inlier_values = values
    outlier_values = values[(values < low_fence) | (values > high_fence)]

    return {
        "q1": float(q1),
        "med": float(median),
        "median": float(median),
        "q3": float(q3),
        "whislo": float(np.min(inlier_values)),
        "whishi": float(np.max(inlier_values)),
        "fliers": outlier_values.astype(float).tolist(),
        "n_obs": int(len(valid)),
        "weight_sum": float(weights.sum()),
    }


def build_plot_frame(combined: pd.DataFrame) -> pd.DataFrame:
    plot_frame = combined.copy()
    plot_frame["plot_value"] = np.log(plot_frame["income"].astype(float))
    return plot_frame


def _ordered_categories(variable_frame: pd.DataFrame) -> pd.DataFrame:
    categories = variable_frame.loc[:, ["category_code", "category_label"]].drop_duplicates().copy()
    categories["sort_key"] = categories["category_code"].map(_sortable_category_code)
    categories = categories.sort_values(["sort_key", "category_code"], kind="stable").reset_index(drop=True)
    return categories


def _display_category_label(variable_name: str, category_code: str, category_label: str) -> str:
    display_label = VIOLIN_DISPLAY_LABELS.get(variable_name, {}).get(str(category_code), category_label)
    wrap_width = VIOLIN_X_LABEL_WRAP_WIDTH.get(variable_name, 18)
    return "\n".join(textwrap.wrap(str(display_label), width=wrap_width, break_long_words=False))


def _weighted_kde(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
    n_points: int = KDE_GRID_POINTS,
    bandwidth_scale: float = KDE_BANDWIDTH_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    if upper <= lower:
        return np.empty(0), np.empty(0)
    mask = (
        np.isfinite(values) & np.isfinite(weights)
        & (weights > 0) & (values >= lower) & (values <= upper)
    )
    v = values[mask]
    w = weights[mask]
    if v.size < 2:
        return np.empty(0), np.empty(0)
    w_norm = w / w.mean()
    kde = gaussian_kde(v, weights=w_norm, bw_method="scott")
    kde.set_bandwidth(kde.factor * bandwidth_scale)
    xs = np.linspace(lower, upper, n_points)
    densities = kde(xs)
    peak = float(densities.max())
    if peak == 0:
        return np.empty(0), np.empty(0)
    densities = densities / peak
    return xs, densities


def _draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    center_x: float,
    width: float,
    lower: float,
    upper: float,
    color: str,
    face_alpha: float = VIOLIN_FACE_ALPHA,
    bandwidth_scale: float = KDE_BANDWIDTH_SCALE,
) -> None:
    xs, densities = _weighted_kde(values, weights, lower=lower, upper=upper, bandwidth_scale=bandwidth_scale)
    if xs.size == 0:
        return
    right = center_x + densities * width
    ax.fill_betweenx(xs, center_x, right, facecolor=to_rgba(color, face_alpha), edgecolor="none", zorder=2)
    ax.plot(right, xs, color=to_rgba(color, VIOLIN_EDGE_ALPHA), linewidth=VIOLIN_EDGE_WIDTH, zorder=3)
    ax.plot([center_x, right[-1]], [xs[-1], xs[-1]], color=to_rgba(color, VIOLIN_EDGE_ALPHA), linewidth=VIOLIN_EDGE_WIDTH, zorder=3)
    ax.plot([center_x, right[0]], [xs[0], xs[0]], color=to_rgba(color, VIOLIN_EDGE_ALPHA), linewidth=VIOLIN_EDGE_WIDTH, zorder=3)


def _collect_variable_groups(
    variable_frame: pd.DataFrame,
    *,
    variable_name: str,
) -> tuple[pd.DataFrame | None, dict[tuple[str, str], dict]]:
    excluded = EXCLUDED_CATEGORY_CODES.get(variable_name, set())
    df = variable_frame
    if excluded:
        df = df.loc[~df["category_code"].astype(str).isin(excluded)]
    if df.empty:
        return None, {}
    df = df.copy()
    df["category_code"] = df["category_code"].astype(str)
    categories = _ordered_categories(df)
    groups: dict[tuple[str, str], dict] = {}
    for (category_code, dataset_name), sub in df.groupby(
        ["category_code", "dataset_name"], sort=False, dropna=False
    ):
        if sub.empty:
            continue
        stats = compute_weighted_box_stats(sub)
        groups[(str(category_code), str(dataset_name))] = {
            "values": sub["plot_value"].to_numpy(dtype=float),
            "weights": sub["weight"].to_numpy(dtype=float),
            "stats": stats,
        }
    return categories, groups


def _build_scope_data(
    scope_frame: pd.DataFrame,
    *,
    variables: list[str],
) -> dict[str, dict]:
    scope_data: dict[str, dict] = {}
    for variable_name in variables:
        variable_frame = scope_frame.loc[scope_frame["variable_name"].eq(variable_name)]
        if variable_frame.empty:
            continue
        categories, groups = _collect_variable_groups(variable_frame, variable_name=variable_name)
        if categories is None or not groups:
            continue
        present = {dataset_name for (_, dataset_name) in groups.keys()}
        datasets = [d for d in DATASET_PLOT_ORDER if d in present]
        scope_data[variable_name] = {
            "categories": categories,
            "groups": groups,
            "datasets": datasets,
        }
    return scope_data


def _build_scope_cache(
    plot_frame: pd.DataFrame,
    *,
    variables: list[str],
    scopes: list[str],
) -> dict[str, dict[str, dict]]:
    return {
        scope_name: _build_scope_data(
            plot_frame.loc[plot_frame["scope_name"].eq(scope_name)],
            variables=variables,
        )
        for scope_name in scopes
    }


def _draw_variable_panel_from_data(
    ax: plt.Axes,
    variable_data: dict,
    *,
    variable_name: str,
    show_ylabel: bool,
    show_variable_title: bool = True,
    y_limits: tuple[float, float] | None = None,
    bandwidth_scale: float = KDE_BANDWIDTH_SCALE,
) -> None:
    categories = variable_data["categories"]
    datasets = variable_data["datasets"]
    groups = variable_data["groups"]
    centers = layout.category_centers(len(categories), gap=CATEGORY_GAP)
    offsets = layout.dataset_offsets(len(datasets), slot_width=SLOT_WIDTH)

    box_stats_list = []
    box_positions = []
    box_datasets = []

    for category_index, category_row in categories.iterrows():
        category_code = str(category_row["category_code"])
        for offset, dataset_name in zip(offsets, datasets):
            entry = groups.get((category_code, dataset_name))
            if entry is None:
                continue
            stats = dict(entry["stats"])
            stats["label"] = ""
            box_stats_list.append(stats)
            position = float(centers[category_index] + offset)
            box_positions.append(position)
            box_datasets.append(dataset_name)

            _draw_half_violin(
                ax, entry["values"], entry["weights"],
                center_x=position,
                width=VIOLIN_MAX_WIDTH,
                lower=stats["whislo"], upper=stats["whishi"],
                color=DATASET_COLORS[dataset_name],
                bandwidth_scale=bandwidth_scale,
            )

    if box_stats_list:
        artists = ax.bxp(
            box_stats_list, positions=box_positions, widths=BOX_DRAW_WIDTH,
            patch_artist=True, showfliers=False,
        )
        for patch, dataset_name in zip(artists["boxes"], box_datasets):
            patch.set_facecolor(to_rgba(DATASET_COLORS[dataset_name], alpha=BOX_FACE_ALPHA))
            patch.set_edgecolor(BOX_EDGE_COLOR)
            patch.set_linewidth(1.0)
        for element_name in ("medians", "whiskers", "caps"):
            for artist in artists[element_name]:
                artist.set_color("#2D3436")
                artist.set_linewidth(1.2 if element_name == "medians" else 1.0)

    display_labels = [
        _display_category_label(variable_name, str(row["category_code"]), row["category_label"])
        for _, row in categories.iterrows()
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels(display_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_AXIS_LABEL if show_ylabel else "")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_variable_title:
        ax.text(
            0.0, 1.02,
            VARIABLE_TITLES.get(variable_name, variable_name.title()),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=16, fontweight="bold",
        )


def _shared_ylimits_from_cache(
    scope_cache: dict[str, dict[str, dict]],
) -> tuple[float, float]:
    lows: list[float] = []
    highs: list[float] = []
    for scope_data in scope_cache.values():
        for variable_data in scope_data.values():
            for entry in variable_data["groups"].values():
                lows.append(entry["stats"]["whislo"])
                highs.append(entry["stats"]["whishi"])
    if not lows:
        return (0.0, 1.0)
    y_min = min(lows)
    y_max = max(highs)
    margin = (y_max - y_min) * 0.04
    return (y_min - margin, y_max + margin)


def _plot_scope_panel_from_data(
    scope_data: dict[str, dict],
    *,
    scope_name: str,
    variables: list[str],
    output_path: Path,
    y_limits: tuple[float, float] | None,
    bandwidth_scale: float = KDE_BANDWIDTH_SCALE,
) -> Path:
    figure_width = max(FIGURE_WIDTH_PER_VARIABLE * len(variables), 24.0)
    fig, axes = plt.subplots(1, len(variables), figsize=(figure_width, FIGURE_HEIGHT), sharey=False)
    axes_flat = np.atleast_1d(axes).flatten()

    for axis, variable_name in zip(axes_flat, variables):
        variable_data = scope_data.get(variable_name)
        if variable_data is None:
            raise ValueError(f"{variable_name} has no plottable data.")
        _draw_variable_panel_from_data(
            axis, variable_data,
            variable_name=variable_name,
            show_ylabel=variable_name == variables[0],
            show_variable_title=True,
            y_limits=y_limits,
            bandwidth_scale=bandwidth_scale,
        )

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=to_rgba(DATASET_COLORS[d], alpha=BOX_FACE_ALPHA),
            edgecolor=BOX_EDGE_COLOR,
        )
        for d in DATASET_PLOT_ORDER
    ]
    legend_labels = [LEGEND_LABELS.get(d, DATASET_LABELS[d]) for d in DATASET_PLOT_ORDER]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        prop={"size": LEGEND_FONT_SIZE},
    )
    fig.suptitle(scope_name.title(), x=0.055, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=LAYOUT_RECT, w_pad=LAYOUT_W_PAD)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _summary_rows_from_cache(
    scope_cache: dict[str, dict[str, dict]],
) -> list[dict]:
    rows: list[dict] = []
    for scope_name, scope_data in scope_cache.items():
        for variable_name, variable_data in scope_data.items():
            categories = variable_data["categories"]
            groups = variable_data["groups"]
            for _, cat_row in categories.iterrows():
                cat_code = str(cat_row["category_code"])
                for dataset_name in DATASET_PLOT_ORDER:
                    entry = groups.get((cat_code, dataset_name))
                    if entry is None:
                        continue
                    stats = entry["stats"]
                    rows.append({
                        "scope_name": scope_name,
                        "variable_name": variable_name,
                        "category_code": cat_code,
                        "category_label": cat_row["category_label"],
                        "dataset_name": dataset_name,
                        "n_obs": stats["n_obs"],
                        "weight_sum": stats["weight_sum"],
                        "whislo": stats["whislo"],
                        "whishi": stats["whishi"],
                    })
    return rows


def run_pipeline(
    synthetic_dir: Path | None = None,
    cfps_path: Path = CFPS_PATH,
    chip_path: Path = CHIP_PATH,
    output_dir: Path | None = None,
    scopes: list[str] | None = None,
    bandwidth_scale: float = KDE_BANDWIDTH_SCALE,
) -> dict[str, object]:
    variables = DEFAULT_PANEL_VARIABLES
    scopes = SCOPE_ORDER if scopes is None else scopes

    synthetic_dir = SYNTHETIC_DIR if synthetic_dir is None else Path(synthetic_dir)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    ensure_directory(output_dir)

    synthetic_frame = prepare_synthetic.load_synthetic_dataset(synthetic_dir)
    cfps_frame = prepare_cfps.load_cfps_dataset(cfps_path)
    chip_frame = prepare_common.load_source_dataset(
        chip_path,
        dataset_name="chip2018",
        columns=CHIP_REQUIRED_COLUMNS,
        income_column="hybrid_annual_wage",
        weight_column=None,
        scope_column="U_R",
        scope_map={"1": "urban", "2": "rural"},
        ownership_column="company_ownership",
        education_column="C_EDU_WORKER",
        occupation_column="C_OCCUPATION",
        ownership_values={"1", "2"},
    )
    combined = pd.concat([synthetic_frame, cfps_frame, chip_frame], ignore_index=True)

    output_paths: dict[str, str] = {}
    plot_frame = build_plot_frame(combined)
    scope_cache = _build_scope_cache(plot_frame, variables=variables, scopes=scopes)
    y_limits = _shared_ylimits_from_cache(scope_cache)
    summary_rows = _summary_rows_from_cache(scope_cache)
    for scope_name in scopes:
        scope_data = scope_cache.get(scope_name, {})
        out_path = output_dir / f"{scope_name}_{LOG_OUTPUT_STEM}_boxplot_violin.png"
        _plot_scope_panel_from_data(
            scope_data,
            scope_name=scope_name,
            variables=variables,
            output_path=out_path,
            y_limits=y_limits,
            bandwidth_scale=bandwidth_scale,
        )
        output_paths[f"{scope_name}_{LOG_OUTPUT_STEM}"] = str(out_path)

    summary_path = output_dir / "violin_panel_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    return {
        "output_dir": str(output_dir),
        "value_scale": LOG_OUTPUT_STEM,
        "scopes": scopes,
        "variables": variables,
        "outputs": output_paths,
        "summary": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Violin + box panel for bar-chart variables.")
    parser.add_argument("--synthetic-dir", type=Path, default=None)
    parser.add_argument("--cfps-path", type=Path, default=CFPS_PATH)
    parser.add_argument("--chip-path", type=Path, default=CHIP_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scopes", nargs="*", default=None, choices=SCOPE_ORDER)
    parser.add_argument("--bandwidth-scale", type=float, default=KDE_BANDWIDTH_SCALE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        synthetic_dir=args.synthetic_dir,
        cfps_path=args.cfps_path,
        chip_path=args.chip_path,
        output_dir=args.output_dir,
        scopes=args.scopes,
        bandwidth_scale=args.bandwidth_scale,
    )


if __name__ == "__main__":
    main()
