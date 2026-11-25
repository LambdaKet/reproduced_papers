"""Values of the paper used for comparison, with their source.

Figure values are our readings of the published figures (arXiv v1 source).
Fig. 3b is read from its drawing commands by ``read_figure_3b.py`` (exact to
0.001 in fidelity and one round). Suppl. Fig. 9 (vector) was read by eye at
high resolution: about +/-0.01 in fidelity and +/-5 % in rounds; the raster
Fig. 2c about +/-0.01-0.02. Text values are quoted from the paper.
"""

from __future__ import annotations

# Fig. 2c (raster): mean fidelity at round 400 and first round the mean reaches
# 0.9, per phase-noise level sigma (in units of pi); 1500 counts, 100 runs.
PHASE_NOISE_FINAL = {0.01: 1.00, 0.02: 0.99, 0.03: 0.97, 0.04: 0.92, 0.05: 0.78}
PHASE_NOISE_ROUND_TO_0P9 = {0.01: 59, 0.02: 79, 0.03: 159, 0.04: 245, 0.05: None}

# Fig. 3b (vector, output of read_figure_3b.py): fidelity after convergence and
# convergence round, per number of stuck generator phase shifters out of 30;
# sigma = 0.02 pi, 1500 counts.
DEFECT_STUCK = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
DEFECT_FIDELITY = [
    0.992, 0.984, 0.985, 0.973, 0.974, 0.965, 0.949,
    0.918, 0.872, 0.803, 0.712, 0.651, 0.456,
]  # fmt: skip
DEFECT_ROUNDS = [380, 416, 438, 510, 557, 602, 674, 645, 621, 600, 468, 377, 302]

# Suppl. Fig. 9b (vector, square-root round axis): sigma = 0.
SHOT_COUNTS = [100, 200, 300, 400, 500, 1000, 1500, 2000]
SHOT_FIDELITY = [0.921, 0.987, 0.990, 0.989, 0.989, 0.986, 0.989, 0.990]
SHOT_ROUNDS = [1873, 997, 429, 231, 152, 84, 68, 58]

# Suppl. Fig. 9a (vector): mean total variation distance at the two ends.
TVD_ENDS = {100: 0.155, 10000: 0.016}

# Text values (main text and Suppl. Sec. 6).
CHIP_START_FIDELITY = 0.1696  # Fig. 2b, chip run: fidelity rises "from 16.96%"
CHIP_INTACT_FIDELITY = 0.9602  # Fig. 2b, after 150 rounds, one chip run
CHIP_DEFECTIVE_FIDELITY = 0.9211  # Fig. 3a, after ~160 rounds, 16/30 stuck
SINGLE_DEFECT = (0.9833, 0.0700)  # mean, std over 100 simulations
NO_DEFECT = (0.9914, 0.0123)
