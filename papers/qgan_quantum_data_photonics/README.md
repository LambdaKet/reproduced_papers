# Quantum Generative Adversarial Learning in Photonics - Reproduction

Status: scaffold only. The reproduction has not started; no results are
available yet.

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

To be defined in the planning phase.

## Install and How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the repo-level `implementation.py`. The CLI is described in
`cli.json`.

```bash
# From inside papers/qgan_quantum_data_photonics
python ../../implementation.py --help

# From the repo root
python implementation.py --paper qgan_quantum_data_photonics --help
```

Global flags (`--config`, `--seed`, `--dtype`, `--device`, `--log-level`,
`--outdir`) are injected by the shared runtime; paper-specific flags are listed
in `cli.json`.

```bash
# From the repo root
python implementation.py --paper qgan_quantum_data_photonics --config configs/example.json
```

### Output directory

Each run writes a timestamped folder under the base `outdir` (default:
`outdir/`):

```
outdir/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
├── run.log                # Run log
└── done.txt               # Scaffold artifact, replaced by real outputs later
```

## Configuration

- `configs/defaults.json`: default parameter values.
- `configs/example.json`: example overlay merged over the defaults.
- `cli.json`: paper-specific CLI arguments and the config keys they change.

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
