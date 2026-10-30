# Sweep plan — `table1_corrected`, attempt 2

Written 2026-10-21 before launch. The launch time is recorded in the sweep's `sweep_status.json` (`started_at`); this file's modification time must precede it.

- **Reason for attempt 2:** attempt 1 (`outdir/sweep_table1_corrected_20261020-174135`) ran with a Framingham C-grid cap whose plan text had no independent pre-launch evidence. This attempt re-runs the whole sweep under this plan.
- **Purpose:** corrected protocol (oversampling only in fitting sets, scaler and kernel PCA fitted on training folds, hyperparameters by inner 5-fold CV) for the paper's methods, the exact references and tuned classical RBF / polynomial SVMs.
- **Candidates:** `configs/accuracy_framingham_corrected.json`, `configs/accuracy_framingham_classical.json`, `configs/accuracy_parkinsons_corrected.json`, `configs/accuracy_parkinsons_classical.json`, `configs/accuracy_wine_corrected.json`, `configs/accuracy_wine_classical.json`.
- **Seeds:** 0, 1, 2, 3, 4. **Expected runs:** 30.
- **Fixed configuration and grids:** outer and inner stratified 5-fold CV (`random_state = 0`); C grid {0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100} for Wine and Parkinson's; **C grid capped at 10 for Framingham** ({0.01, …, 10}, 7 values) because a precomputed linear-kernel SVC takes 112.5 s per fit at C = 100 on a 3,000-row fold (0.1 s at C = 1); RBF γ multipliers {0.01, 0.1, 1, 10, 100} × 1 / (d · Var X); polynomial degree {2, 3}, coef0 {0, 1}; kernel-PCA components 2–7; 1,024 shots for swap-test kernels.
- **Metric:** outer-fold accuracy (balanced accuracy secondary); direction: maximise; inner selection by mean inner-CV accuracy, first grid point on ties; checkpoint: not applicable.
- **Aggregation:** mean over seeds of the 5-fold mean; std over seeds (ddof = 1); mean fold std reported. No cross-candidate selection.
- **Expected outcome:** identical metrics to attempt 1 (deterministic given seeds).
- **Code state:** commit `615a02e` plus uncommitted paper folder, `paper_tree_sha256 = 8d5da9f7d58a7804815412caa45324970a694e63736a8d97208a2800e2c29ad1`.
- **Resources:** local CPU, 5 workers; attempt 2 of 2; a failed run makes the sweep PARTIAL.
- **Command:** `.venv/bin/python papers/distributed_secure_kernels/utils/run_sweep.py --sweep-id table1_corrected --seeds 0 1 2 3 4 --workers 5 --configs configs/accuracy_framingham_corrected.json configs/accuracy_framingham_classical.json configs/accuracy_parkinsons_corrected.json configs/accuracy_parkinsons_classical.json configs/accuracy_wine_corrected.json configs/accuracy_wine_classical.json`
