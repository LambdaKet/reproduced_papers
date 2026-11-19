"""The stored Fig. 3b values are the output of the figure reader."""

from __future__ import annotations

import sys

import pytest
from common import PROJECT_DIR, REPO_ROOT

sys.path.insert(0, str(PROJECT_DIR / "utils"))

import paper_values as paper  # noqa: E402
import read_figure_3b  # noqa: E402

# Unpacked arXiv source archive (https://arxiv.org/e-print/2310.00585v1).
FIG3 = REPO_ROOT / "data" / "qgan_quantum_data_photonics" / "sources"
FIG3 = FIG3 / "arxiv_2310.00585v1_src" / "fig3.pdf"


@pytest.mark.skipif(
    not FIG3.exists(), reason="arXiv source of the paper not downloaded"
)
def test_figure_3b_values_match_the_reading():
    values = read_figure_3b.read(FIG3)
    assert [round(f, 3) for f in values["fidelity"]] == paper.DEFECT_FIDELITY
    assert [round(r) for r in values["rounds"]] == paper.DEFECT_ROUNDS
    # The no-defect marker agrees with the 99.14 % quoted in the text.
    assert abs(values["fidelity"][0] - paper.NO_DEFECT[0]) < 0.001
