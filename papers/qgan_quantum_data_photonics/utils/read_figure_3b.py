"""Read the values of the paper's Fig. 3b from its vector source.

Fig. 3b of arXiv:2310.00585v1 is drawn as vectors in ``fig3.pdf``, inside the
arXiv source archive (https://arxiv.org/e-print/2310.00585v1). This script
reads the centres of the markers and the positions of the major axis ticks
from the drawing commands, then converts each marker to a value with a linear
fit of the ticks (the fit is exact to 0.0005 in fidelity and 0.5 in rounds).
``DEFECT_FIDELITY`` and ``DEFECT_ROUNDS`` in ``paper_values.py`` are its
output, rounded to 0.001 and to one round.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/read_figure_3b.py \\
        path/to/arxiv_2310.00585v1_src/fig3.pdf [--check]

``--check`` exits with status 1 if ``paper_values.py`` differs from the reading.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import paper_values as paper

# Fill colours (CMYK) of the two series, from the legend of Fig. 3b.
FIDELITY_RED = (0.312, 0.984, 0.863, 0.005)  # "Fidelity after convergence", circles
ROUNDS_BLUE = (0.955, 0.781, 0.279, 0.002)  # "Convergence round", squares
# Panel (b) occupies y < 105 pt (panel (a) and the legend are above).
PANEL_B_TOP = 105.0
LEFT_TICKS = (0.2, 0.4, 0.6, 0.8, 1.0)  # fidelity axis
RIGHT_TICKS = (0, 200, 400, 600, 800, 1000)  # convergence-round axis

_TOKEN = re.compile(
    r"\((?:\\.|[^\\)])*\)|\[[^\]]*\]|/[^\s/\[\]()<>]+|[-+]?\d*\.?\d+|[A-Za-z*]+"
)
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def _content(pdf: Path) -> str:
    """Return the page content stream (the largest compressed stream)."""
    raw = pdf.read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S)
    decoded = []
    for s in streams:
        try:
            decoded.append(zlib.decompress(s))
        except zlib.error:
            continue
    return max(decoded, key=len).decode("latin-1")


def _paths(content: str) -> list[tuple[str, tuple, np.ndarray]]:
    """Interpret path operators; return (kind, colour, points) per subpath.

    Points are in PDF points after the transformation matrix; kind is "fill"
    or "stroke". Only the operators used by this figure are handled.
    """
    ctm, saved, operands, subpaths, out = np.eye(3), [], [], [], []
    fill = stroke = None

    def point(x: float, y: float) -> tuple[float, float]:
        v = np.array([x, y, 1.0]) @ ctm
        return float(v[0]), float(v[1])

    for token in _TOKEN.findall(content):
        if _NUMBER.fullmatch(token):
            operands.append(float(token))
            continue
        if token[0] in "([/":
            operands.append(token)
            continue
        if token == "q":
            saved.append(ctm.copy())
        elif token == "Q":
            ctm = saved.pop()
        elif token == "cm":
            a, b, c, d, e, f = operands[-6:]
            ctm = np.array([[a, b, 0], [c, d, 0], [e, f, 1]]) @ ctm
        elif token == "m":
            subpaths.append([point(*operands[-2:])])
        elif token in ("l", "c"):
            subpaths[-1].append(point(*operands[-2:]))
        elif token == "re":
            x, y, w, h = operands[-4:]
            subpaths.append(
                [point(x, y), point(x + w, y), point(x + w, y + h), point(x, y + h)]
            )
        elif token in ("S", "s", "f", "f*", "F", "B", "B*", "b", "b*", "n"):
            kind = "stroke" if token in ("S", "s") else "fill"
            colour = stroke if kind == "stroke" else fill
            if token != "n":
                out.extend((kind, colour, np.array(p)) for p in subpaths)
            subpaths = []
        elif token == "k":
            fill = tuple(round(v, 3) for v in operands[-4:])
        elif token == "K":
            stroke = tuple(round(v, 3) for v in operands[-4:])
        operands = []
    return out


def _centre(points: np.ndarray) -> tuple[float, float]:
    return (
        (points[:, 0].min() + points[:, 0].max()) / 2,
        (points[:, 1].min() + points[:, 1].max()) / 2,
    )


def _markers(paths, colour: tuple, n_points: int) -> list[tuple[float, float]]:
    centres = [
        _centre(p)
        for kind, c, p in paths
        if kind == "fill" and c == colour and len(p) == n_points
    ]
    return sorted(c for c in centres if c[1] < PANEL_B_TOP)


def _major_ticks(paths, x_min: float, x_max: float) -> list[float]:
    """Heights of the major ticks (short horizontal strokes longer than 1.5 pt)."""
    heights = set()
    for kind, _, p in paths:
        if kind != "stroke" or len(p) != 2 or abs(p[0, 1] - p[1, 1]) > 0.01:
            continue
        length = abs(p[0, 0] - p[1, 0])
        if x_min < p[:, 0].min() and p[:, 0].max() < x_max and 1.5 < length < 5:
            if p[0, 1] < PANEL_B_TOP:
                heights.add(round(float(p[0, 1]), 3))
    return sorted(heights)


def _scale(heights: list[float], values: tuple) -> np.poly1d:
    if len(heights) != len(values):
        raise ValueError(f"expected {len(values)} major ticks, found {heights}")
    return np.poly1d(np.polyfit(heights, values, 1))


def read(pdf: Path) -> dict[str, list[float]]:
    """Fidelity and convergence round per marker, ordered by defect rate."""
    paths = _paths(_content(pdf))
    fidelity = _markers(paths, FIDELITY_RED, 5)  # circle: move + 4 curves
    rounds = _markers(paths, ROUNDS_BLUE, 4)  # square
    n = len(paper.DEFECT_STUCK)
    if len(fidelity) != n or len(rounds) != n:
        raise ValueError(
            f"expected {n} markers per series, found {len(fidelity)}, {len(rounds)}"
        )
    left = _scale(_major_ticks(paths, 25, 35), LEFT_TICKS)
    right = _scale(_major_ticks(paths, 195, 205), RIGHT_TICKS)
    return {
        "fidelity": [float(left(y)) for _, y in fidelity],
        "rounds": [float(right(y)) for _, y in rounds],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path, help="fig3.pdf from the arXiv source archive")
    parser.add_argument(
        "--check", action="store_true", help="compare with paper_values.py"
    )
    args = parser.parse_args()
    values = read(args.pdf)
    print("| Stuck / 30 | Fidelity | Round | Stored fidelity | Stored round |")
    print("|---:|---:|---:|---:|---:|")
    for k, f, r, sf, sr in zip(
        paper.DEFECT_STUCK, values["fidelity"], values["rounds"],
        paper.DEFECT_FIDELITY, paper.DEFECT_ROUNDS, strict=True,
    ):  # fmt: skip
        print(f"| {k} | {f:.4f} | {r:.1f} | {sf:.3f} | {sr} |")
    if not args.check:
        return 0
    same = [round(f, 3) for f in values["fidelity"]] == paper.DEFECT_FIDELITY and [
        round(r) for r in values["rounds"]
    ] == paper.DEFECT_ROUNDS
    print("paper_values.py matches the reading" if same else "paper_values.py differs")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
