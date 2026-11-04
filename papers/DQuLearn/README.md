# Distributed Quantum Learning with co-Management in a Multi-tenant Quantum System - Reproduction

Status: implementation done (phase 02): runnable, tested code and smoke runs; no experiment results yet (phase 03).

## Reference and Attribution

- Paper: "Distributed Quantum Learning with co-Management in a Multi-tenant Quantum System" (IEEE BigData 2023)
- Authors: Anthony D'Onofrio Jr., Amir Hossain, Lesther Santana, Naseem Machlovi, Samuel Stein, Jinwei Liu, Ang Li, Ying Mao
- arXiv: [2312.08158](https://arxiv.org/abs/2312.08158) (v1, 13 Dec 2023; cs.DC)
- Original repository: none linked by the paper. The paper adopts its circuits from the open-source QuClassi code (Stein et al., "QuClassi: A hybrid deep neural network architecture based on quantum state fidelity", MLSys 2022) and gives no URL for it.
- License and attribution notes: cite the paper above when reusing this reproduction.

## Overview

DQuLearn trains a hybrid quantum-classical classifier on several small quantum machines instead of one large one. Its workload is QuClassi, a classifier based on quantum-state fidelity; parameter-shift gradients turn each training epoch into a bank of independent circuits. A co-Manager sends each circuit to a registered worker that has more free qubits than the circuit needs and the lowest classical resource usage. The paper reports accuracy on four MNIST digit pairs, the runtime of one epoch with 1, 2 and 4 workers, and the gain from serving four clients at once (multi-tenancy), with simulators as workers.

This reproduction runs the paper's QuClassi workload (ported from QuClassi's released code to Qiskit 2.x) through a local co-Manager (Algorithm 2) and worker processes reached over RPyC, plus fair execution baselines; the experiments are run in the next phase.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Main entry point: the repo-level `implementation.py`. The CLI is entirely described in `cli.json`, so updating/adding arguments does not require editing Python code.

```bash
# From inside papers/DQuLearn
python ../../implementation.py --help

# From the repo root
python implementation.py --paper DQuLearn --help
```

Main flags (see `cli.json` for the authoritative list): `--config PATH`, `--seed INT`, `--qubits`, `--layers`, `--workers`, `--worker-qubits`, `--backend` (`manager`, `pool`, `batched`, `local`), `--mode` (multi-tenant: `multi`, `single`, `sequential`), `--policy` (`least_cpu`, `round_robin`), `--compile-mode`, `--pair`, `--lr`, `--epochs`, `--update-rule`, `--shift-schedule`, `--circuits-per-epoch`, `--exact`. The device must be `cpu`.

Example runs, from the repository root:

```bash
# Smoke run (defaults): one 80-circuit epoch through the co-Manager and 2 workers
python implementation.py --paper DQuLearn

# Worker scaling at the paper's circuit count (Fig. 5): 4 workers, 2 layers
python implementation.py --paper DQuLearn --config configs/worker_scaling_original.json --workers 4 --layers 2

# Multi-tenant (Fig. 6): four concurrent clients on workers of 5, 10, 15 and 20 qubits
python implementation.py --paper DQuLearn --config configs/multi_tenant_original.json

# Accuracy on a digit pair (Sec. IV-B), 2 workers, exact probabilities
python implementation.py --paper DQuLearn --config configs/accuracy_original.json --pair 3,8
```

From inside `papers/DQuLearn`, use `python ../../implementation.py` with the same flags.

The script saves a snapshot of the resolved config alongside results and logs.

### Data

The paper uses MNIST and links its original page, which no longer serves the files. This reproduction uses the copy that QuClassi's released code loads (`tf.keras.datasets.mnist.load_data()`): the full training (60,000 images) and test (10,000 images) splits, identical to the original MNIST files and in the same order (checked against torchvision's copies). MNIST is copyright Yann LeCun and Corinna Cortes, under the CC BY-SA 3.0 licence (as stated in the Keras loader). Download it from the repository root (11.5 MB, git-ignored):

```bash
mkdir -p data/DQuLearn/MNIST
curl -fsSL -o data/DQuLearn/MNIST/mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
shasum -a 256 data/DQuLearn/MNIST/mnist.npz  # expect 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
```

### Data location and shared code

- The shared data root defaults to the repository's `data/` folder; each paper is scoped to a subfolder named after the paper, here `data/DQuLearn/`. The runner resolves this via `runtime_lib.data_paths.paper_data_dir`, so you usually just pass `--paper` or set `dataset.root` to `null` and downloads land in the right place.
- Override the base root with `DATA_DIR=/abs/path` or `--data-root /abs/path` when calling `implementation.py`; the paper subfolder is appended automatically.
- Reusable helpers can live under `papers/shared/`; import them from the paper code to avoid duplication.

### Output directory and generated files

At each run, a timestamped folder is created under the base `outdir` (default: `outdir`):

```
outdir/run_YYYYMMDD-HHMMSS/
├── config_snapshot.json   # Resolved configuration used for the run
├── run.log                # Lifecycle, progress and evaluation events
├── run_status.json        # Run status, code and data identity, evidence hashes
├── metrics.json           # Structured results
└── worker-<i>.log         # One log per worker process (manager backend)
```

Notes:
- Change the base output directory with `--outdir` or in `configs/defaults.json` (key `outdir`).
- `outdir/` is ignored by Git (repository `.gitignore`).

## Configuration

Place configuration files in `configs/`.

- `defaults.json`: a lightweight smoke run; every named config overlays it. Named configs: `worker_scaling_original.json` and its `_pool_baseline` and `_batched_baseline` variants; `multi_tenant_original.json`, `multi_tenant_single_tenant.json`, `multi_tenant_sequential_jobs.json`, `multi_tenant_round_robin.json`; `accuracy_original.json`, `accuracy_quclassi_reference.json`, `accuracy_classical.json`, `accuracy_shots.json`; smoke sizes `multi_tenant_reduced.json` and `accuracy_reduced.json`. Each file's `description` says what it runs.
- `cli.json` declares the CLI arguments, their types, and the config keys they mutate.
- The shared runtime automatically loads `configs/defaults.json`, `cli.json`, and calls `lib/runner.py::train_and_evaluate`, so keep that entry point in every project.
- Any config value named `dtype` is normalized into a `(label, torch.dtype)` pair at runtime via `runtime_lib.dtypes`, so you can inspect both the requested string and the resolved `torch.dtype` without duplicating conversion logic.
- Keys typically include: dataset, model, training, evaluation, logging.
- Precision control: add a top-level `"dtype"` entry (mirrors `"seed"`) to force the entire pipeline (datasets + models) to run in a specific torch dtype; individual models may still expose a `model.dtype` override if required.

**Placeholder guard:** If any config value still contains a `<<...>>` placeholder (e.g., `"teacher_path": "<<TEACHER_PATH>>"`), the shared runner aborts early with a clear error. Replace these placeholders with real paths/values before launching a run.

## Results and Analysis

No results yet.

## Extensions and Next Steps

None planned yet.

## Reproducibility Notes

Library versions are pinned in `requirements.txt`. `--seed` sets the subsample, the validation split, the initial parameters and every circuit's simulator seed, so a circuit gives the same counts on any worker.

## Testing

Run tests from inside the `papers/DQuLearn/` directory:

```bash
cd papers/DQuLearn
pytest -q
```

Notes:
- Tests are scoped to this paper folder and expect the current working directory to be `DQuLearn/`.
- The tests check the circuits against QuClassi's layout and an independent fidelity computation, the co-Manager rules of Algorithm 2, the training updates, the data loader, and that training through two worker processes is identical to local training.
- If `pytest` is not installed: `pip install pytest`.
