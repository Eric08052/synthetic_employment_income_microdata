import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple


DARK_GRAY = "#2C3E50"
LIGHT_GRAY = "#ECF0F1"
MEDIUM_GRAY = "#95A5A6"

REGION_SCATTER_COLORS = {
    "East": "#E89A9C",
    "Central": "#EFCB9A",
    "West": "#B7C9C0",
}

REGION_SCATTER_ALPHA = 0.72


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "axes.facecolor": "white",
            "axes.edgecolor": DARK_GRAY,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.color": DARK_GRAY,
            "ytick.color": DARK_GRAY,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": LIGHT_GRAY,
            "legend.fancybox": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def clean_axis(ax, show_grid: bool = True, grid_axis: str = "both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK_GRAY)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    if show_grid:
        ax.grid(True, alpha=0.4, linewidth=0.5, color=MEDIUM_GRAY)
        ax.set_axisbelow(True)
        if grid_axis == "x":
            ax.yaxis.grid(False)
        elif grid_axis == "y":
            ax.xaxis.grid(False)
    else:
        ax.grid(False)

    ax.tick_params(colors=DARK_GRAY)


def style_scatter(
    ax,
    x,
    y,
    color: str,
    alpha: float = 0.7,
    size: int = 60,
    edgecolor: str = "white",
    edgewidth: float = 0.8,
    **kwargs,
):
    return ax.scatter(
        x,
        y,
        c=color,
        alpha=alpha,
        s=size,
        edgecolors=edgecolor,
        linewidth=edgewidth,
        **kwargs,
    )


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    if figsize is None:
        figsize = (5 * ncols + 1, 4.5 * nrows + 0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            clean_axis(ax)
    else:
        clean_axis(axes)

    return fig, axes


def plot_validation_urban_scatter(
    merged,
    regression: dict,
    output_path,
    lims: Optional[Tuple[float, float]] = None,
):
    fig, ax = create_figure(1, 1, figsize=(8, 7))
    draw_validation_urban_scatter(
        ax,
        merged,
        regression,
        lims=lims,
        title=None,
        show_ylabel=True,
        show_legend=True,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def draw_validation_urban_scatter(
    ax,
    merged,
    regression: dict,
    lims: Optional[Tuple[float, float]] = None,
    *,
    title: str | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
):
    x = merged["log_ext"].values
    y = merged["log_sim"].values

    for region in ["East", "Central", "West"]:
        mask = merged["region"] == region
        if mask.any():
            style_scatter(
                ax,
                x[mask],
                y[mask],
                color=REGION_SCATTER_COLORS.get(region, "#8AA0B6"),
                alpha=0.7,
                size=50,
                label=region,
            )

    if lims is None:
        lim_lo = min(x.min(), y.min()) - 0.1
        lim_hi = max(x.max(), y.max()) + 0.1
    else:
        lim_lo, lim_hi = lims
    lims = [lim_lo, lim_hi]

    ax.plot(lims, lims, "--", color="#8A93A5", linewidth=1.5, alpha=0.7, label="_nolegend_")

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = regression["alpha"] + regression["beta"] * x_line
    ax.plot(x_line, y_line, "--", color="#40566B", linewidth=3.2, label="_nolegend_")

    ax.set_xlabel("log(annual average wage of urban non-private employees)", fontsize=11)
    ax.set_ylabel("log(synthetic income data)", fontsize=11 if show_ylabel else 0)
    if not show_ylabel:
        ax.set_ylabel("")
    if title is not None:
        ax.set_title(
            title,
            fontsize=12,
            fontweight="medium",
            color=DARK_GRAY,
            pad=10,
        )
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.9)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_validation_rural_scatter(
    merged,
    output_path,
    income_col: str = "mean_income",
    income_col_ext: str = "rural_income",
    shared_lims: Optional[Tuple[float, float]] = None,
    axis_label_mode: str = "linear",
):
    fig, ax = create_figure(figsize=(10, 8))
    draw_validation_rural_scatter(
        ax,
        merged,
        income_col=income_col,
        income_col_ext=income_col_ext,
        shared_lims=shared_lims,
        axis_label_mode=axis_label_mode,
        show_ylabel=True,
        show_legend=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_validation_rural_scatter(
    ax,
    merged,
    income_col: str = "mean_income",
    income_col_ext: str = "rural_income",
    shared_lims: Optional[Tuple[float, float]] = None,
    axis_label_mode: str = "linear",
    *,
    show_ylabel: bool = True,
    show_legend: bool = True,
):
    for region in ["East", "Central", "West"]:
        region_data = merged[merged["region"] == region]
        if len(region_data) > 0:
            ax.scatter(
                region_data[income_col_ext],
                region_data[income_col],
                c=REGION_SCATTER_COLORS[region],
                label=region,
                alpha=REGION_SCATTER_ALPHA,
                s=50,
            )

    valid = merged[[income_col_ext, income_col]].dropna()
    ref_lims = None
    if shared_lims is not None:
        ref_lims = (float(shared_lims[0]), float(shared_lims[1]))
    elif len(valid) > 0:
        lo = min(float(valid[income_col_ext].min()), float(valid[income_col].min()))
        hi = max(float(valid[income_col_ext].max()), float(valid[income_col].max()))
        span = hi - lo
        if span <= 0:
            span = max(1.0, abs(lo) * 0.1)
        pad = span * 0.02
        ref_lims = (lo - pad, hi + pad)

    if ref_lims is not None:
        ax.plot(
            [ref_lims[0], ref_lims[1]],
            [ref_lims[0], ref_lims[1]],
            "--",
            color="#7E8793",
            linewidth=1.6,
            alpha=0.78,
            label="_nolegend_",
        )

    if len(valid) >= 3:
        z = np.polyfit(valid[income_col_ext], valid[income_col], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(valid[income_col_ext].min(), valid[income_col_ext].max(), 100)
        ax.plot(x_line, p_line(x_line), "--", color="#1F1F1F", linewidth=3.0, alpha=0.92, label="_nolegend_")

    if axis_label_mode == "log":
        ax.set_xlabel("log(annual per capita disposable income for rural population)", fontsize=12)
        ax.set_ylabel("log(synthetic income data)", fontsize=12 if show_ylabel else 0)
    else:
        ax.set_xlabel("Annual per capita disposable income in rural areas", fontsize=12)
        ax.set_ylabel("Synthetic data: Annual income of the rural employed population", fontsize=12 if show_ylabel else 0)
    if not show_ylabel:
        ax.set_ylabel("")
    if ref_lims is not None:
        ax.set_xlim(ref_lims[0], ref_lims[1])
        ax.set_ylim(ref_lims[0], ref_lims[1])
    if show_legend:
        ax.legend(loc="upper left")
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    ax.grid(True, alpha=0.3)


setup_style()
