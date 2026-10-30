# Quantum Generative Adversarial Learning in Photonics - Reproduction

Status: implementation ready and tested; no results yet (experiments run in
the next phase).

## Reference and Attribution

- Paper: "Quantum generative adversarial learning in photonics", Optics Letters
  48(20), 5197-5200 (2023)
- Authors: Yizhi Wang, Shichuan Xue, Yaxuan Wang, Yong Liu, Jiangfang Ding,
  Weixu Shi, Dongyang Wang, Yingwen Liu, Xiang Fu, Guangyao Huang, Anqi Huang,
  Mingtang Deng, Junjie Wu
- DOI: [10.1364/OL.505084](https://doi.org/10.1364/OL.505084)
- Open version: [arXiv:2310.00585v1](https://arxiv.org/abs/2310.00585v1)
- Original repository: none released. The paper states that data are available
  from the authors upon reasonable request.
- License and attribution notes: cite the paper above when using this
  reproduction. This folder contains independent code; no code or data from the
  authors is included.

## Original Paper

The paper trains a quantum generative adversarial network (QGAN) on a photonic
chip to learn quantum data: a random maximally entangled state of two
four-level systems (ququarts), path-encoded on 4 + 4 optical modes. The
generator is a pair of universal 4-mode Mach-Zehnder interferometer meshes; the
discriminator is a quantum projective measurement made of two 4-mode networks.
The paper reports the fidelity of the generated state, and studies robustness to
defective phase shifters and to phase noise.

## Reproduction Scope

Everything is simulated: no photonic chip is available. The simulator follows
the chip's circuits as drawn in the supplementary document (Suppl. Figs. 2(d)
and 3): two 15-phase-shifter meshes for the generator and two 6-phase-shifter
projection circuits for the discriminator, the cost d of Eq. 1, parameter-shift
gradients (shift pi/4), Gaussian phase noise on every phase shifter, Poisson
coincidence counts (1500 per measurement setting) and stuck phase shifters.
Planned experiments: the phase-noise sweep (Fig. 2c), the defect-rate sweep
(Fig. 3b), the shot-noise study (Suppl. Fig. 9), the single-defect study
(Suppl. Sec. 6), the two chip runs replayed in simulation (Figs. 2a-b, 3a), and
our control: the best fidelity each circuit can reach.

Deviations and choices, stated because the paper does not give them:

- Training settings: the paper gives no learning rate, number of steps per
  turn or end-of-turn rule. With a fixed number of steps per turn, the game
  oscillates and does not learn: the cost |a - b| has a kink at a = b, where
  the generator keeps taking full-size steps. We let the generator end its turn
  once the measured d falls below a threshold, as the turns of Fig. 2a suggest
  (2-6 steps, ending with a gap of about 0.02-0.04); the step size, steps per
  turn and threshold are calibrated on one condition of Fig. 2c.
- Random choices not stated in the paper: Haar-random targets; uniform
  discriminator starting phases and stuck values; the same number of stuck
  phase shifters in each circuit; new phase noise at every circuit execution.

## Install and How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Quick smoke run

```bash
# From the repo root: 2 targets, 10 rounds, sigma = 0.01 pi, 1500 counts
python implementation.py --paper qgan_quantum_data_photonics
```

### One experiment

Each config in `configs/` runs one condition on a block of target seeds (one
target per seed). Paper-specific flags are listed in `cli.json`; angles are in
units of pi.

```bash
# From the repo root
python implementation.py --paper qgan_quantum_data_photonics \
    --config configs/phase_noise_sweep_original.json --phase-noise-pi 0.04
python implementation.py --paper qgan_quantum_data_photonics \
    --config configs/defect_rate_sweep_original.json --defects-per-circuit 8
```

### Sweeps

`utils/run_sweep.py` runs every candidate of a spec in `utils/sweep_specs/`
one after the other, checks each run's evidence files with the logging
validator given by `--validator`, and writes `sweep_status.json`, `sweep.log`
and `sweep_summary.json` in the sweep folder.

```bash
# From the repo root
python papers/qgan_quantum_data_photonics/utils/run_sweep.py \
    papers/qgan_quantum_data_photonics/utils/sweep_specs/phase_noise_sweep.json \
    --sweep-dir papers/qgan_quantum_data_photonics/outdir/sweep_phase_noise \
    --validator <path to validate_logging.py>
```

### Output directory

Each run writes a timestamped folder under the base `outdir` (default:
`outdir/`):

```
outdir/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration
├── run.log                # Run log
├── run_status.json        # Status, code commit, dataset identity, artifact hashes
├── metrics.json           # Summary metrics (root fidelity, convergence round, d)
└── trajectories.npz       # Per-round, per-seed fidelity and d (ceiling.npz for the ceiling)
```

## Configuration

- `configs/defaults.json`: smoke settings and every key, with the provisional
  training settings (step 0.3 pi, up to 20 steps per turn, generator stop at
  d <= 0.05) until the calibration fixes them.
- `configs/<experiment>_original.json`: one overlay per planned experiment
  (`phase_noise_sweep`, `defect_rate_sweep`, `shot_noise_sweep`,
  `single_defect`, `single_defect_position`, `intact_chip_emulation`,
  `defective_chip_emulation`, `oracle_ceiling`, `oracle_ceiling_chip`,
  `fixed_discriminator`, `calibration`, `minimal_learning`,
  `minimal_learning_noisy`, `shot_noise_tvd`).
- `cli.json`: paper-specific CLI arguments and the config keys they change.
- The simulator runs on CPU only (`--device cpu`).

## Data

Nothing is downloaded. As in the paper, the learning targets are random
maximally entangled two-ququart states generated numerically. Each run draws its
target, the discriminator's initial phases, the stuck phase-shifter values and
the noise from its seed, so no data file is stored. The paper does not state the
target distribution; we draw targets from the Haar measure (our choice), which
covers every maximally entangled two-ququart state.

The authors' measured chip data are not public (available from the authors on
request) and are not used: the chip runs (Fig. 2a-b, Fig. 3a) are emulated in
simulation and compared with the values the paper reports. If these data are
obtained later, they go under
`data/qgan_quantum_data_photonics/authors_measured_data/`.

## Results Obtained and Comparison with the Paper

No results yet.

## Limitations

To be written once results are available.

## Tests

Run tests from inside the paper folder:

```bash
cd papers/qgan_quantum_data_photonics
pytest -q
```
