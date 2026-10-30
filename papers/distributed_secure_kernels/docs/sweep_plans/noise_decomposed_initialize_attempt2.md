# Sweep plan — `noise_decomposed_initialize`, attempt 2

Written 2026-10-21 before launch. The launch time is recorded in the sweep's `sweep_status.json` (`started_at`); this file's modification time must precede it.

- **Nature:** post hoc diagnostic. The hypothesis (the authors' Qiskit decomposed `initialize` into noisy gates) was formed after the Fig. 3 gap was observed. Attempt 1 (`outdir/sweep_noise_decomposed_initialize_20261020-182637`) had no plan dated before launch; this attempt re-runs it under this plan.
- **Purpose:** test whether noisy, decomposed state preparation reproduces the authors' logged Fig. 3 values.
- **Candidates:** `configs/noise_parkinsons_decomposed_initialize.json`, `configs/noise_framingham_decomposed_initialize.json` (distributed kernel at noise levels 1 and 2, `noisy_initialize: true`, 1,024 shots).
- **Seeds:** 0, 1, 2, 3, 4 (one repetition each). **Expected runs:** 10.
- **Fixed configuration:** paper protocol (authors' 80 % subset, oversampling before split, `StandardScaler` on the subset, stratified 5-fold CV with `random_state = 0`, SVM C = 1, kernel PCA with 2–7 components).
- **Metric:** 5-fold CV accuracy (balanced accuracy secondary); direction: maximise; split: CV folds of the evaluated subset; checkpoint: not applicable.
- **Aggregation:** mean over seeds of the 5-fold mean; variability: std over seeds (ddof = 1); mean fold std reported.
- **Selection rule:** none (seed repetitions; `selected_candidates` empty). Tie tolerance: not applicable.
- **Comparison rule:** compare with the authors' logs (Parkinson's kernel SVM 0.7873 ± 0.0655 at 0.1 %, 0.6330 ± 0.0087 at 1 %; Framingham 0.5807 ± 0.0098, 0.5814 ± 0.0072); "reproduced" when |difference| ≤ the authors' fold std.
- **Expected outcome:** identical metrics to attempt 1 (deterministic given seeds).
- **Code state:** commit `615a02e` plus uncommitted paper folder, `paper_tree_sha256 = 8d5da9f7d58a7804815412caa45324970a694e63736a8d97208a2800e2c29ad1`.
- **Resources:** local CPU, 5 workers; at most one more attempt is not available (this is attempt 2 of 2); a failed run makes the sweep PARTIAL.
- **Command:** `.venv/bin/python papers/distributed_secure_kernels/utils/run_sweep.py --sweep-id noise_decomposed_initialize --seeds 0 1 2 3 4 --workers 5 --configs configs/noise_parkinsons_decomposed_initialize.json configs/noise_framingham_decomposed_initialize.json`
