"""Shared look of the reproduction's figures: palette, ink, marks and saving.

Colours follow a fixed categorical order, checked for colour-vision-deficiency
separation of neighbouring series on the light surface. Filled markers are this
reproduction; hollow markers of the same hue are the paper's values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Set the matplotlib defaults shared by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": BASELINE,
            "ytick.color": BASELINE,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
        }
    )


def dots(
    ax, x, y, error, color: str, label: str, hollow: bool = False, marker: str = "o"
):
    """Draw one series as dots with whiskers; ``error`` is (below, above) or None.

    Filled dots carry a surface-coloured ring; hollow dots (paper values) keep
    the series hue as their outline.
    """
    style = (
        {"markerfacecolor": SURFACE, "markeredgecolor": color, "markeredgewidth": 1.8}
        if hollow
        else {
            "markerfacecolor": color,
            "markeredgecolor": SURFACE,
            "markeredgewidth": 1.2,
        }
    )
    return ax.errorbar(
        x,
        y,
        yerr=error,
        fmt=marker,
        markersize=7,
        color=color,
        ecolor=color,
        elinewidth=1.4,
        capsize=0,
        label=label,
        zorder=3,
        **style,
    )


def scope_note(fig, text: str) -> None:
    """Write the result-scope label under the panels."""
    fig.text(
        0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5, color=MUTED, wrap=True
    )


def save(fig, name: str, *folders: Path) -> list[Path]:
    """Save ``<name>.png`` in every folder (the sweep folder and ``results/``)."""
    paths = []
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
