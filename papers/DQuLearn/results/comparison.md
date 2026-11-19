# Phase 5 comparison

Built by `utils/build_comparison.py` from the sweeps' structured metrics; every expected run completed. Local values: one laptop CPU, simulation. Paper values: its text, never ranked with local values. Rule: a paired ratio differs when |mean − 1| > 2 std; 95% Student-t interval in brackets.

## Accuracy kept (claim accuracy-parity)

Test accuracy (%), mean ± std over seeds 0, 1, 2; 4 PCA features (QuClassi's preprocessing), 25 epochs, learning rate chosen on validation. Scope: partial.

| System | Role | Runs | 3/9 | 3/8 | 3/6 | 1/5 | Parameters | Seconds per run |
|---|---|---:|---:|---:|---:|---:|---|---:|
| DQuLearn, qubit model, 2 workers | reproduced method | 12/12 | 95.2 ± 0.7 | 72.6 ± 0.9 | 95.8 ± 0.1 | 93.7 ± 0.7 | 4 per class | 18.3 |
| QuClassi rule, qubit model, one process | paper's baseline (non-distributed) | 12/12 | 94.9 ± 0.7 | 72.1 ± 1.2 | 95.8 ± 0.1 | 93.9 ± 0.5 | 4 per class | 12.6 |
| Logistic regression, same 4 features | added classical baseline | 12/12 | 95.8 ± 0.4 | 75.3 ± 1.1 | 95.1 ± 0.2 | 95.0 ± 2.9 | 5 | 0.3 |
| DQuLearn, photonic model, 2 workers | MerLin adaptation | 12/12 | 82.8 ± 22.1 | 66.0 ± 12.1 | 89.1 ± 11.5 | 89.9 ± 6.1 | 12 per class | 37.9 |
| QuClassi rule, photonic model, one process | MerLin adaptation, non-distributed | 12/12 | 86.9 ± 8.9 | 60.9 ± 11.5 | 81.7 ± 21.3 | 63.6 ± 22.5 | 12 per class | 26.1 |
| Exact photonic shift, 2 workers | MerLin adaptation, exact shift | 12/12 | 81.1 ± 25.0 | 65.9 ± 12.2 | 90.9 ± 8.5 | 83.0 ± 18.3 | 12 per class | 63.3 |
| Paper, DQuLearn with 2 workers | paper-reported | — | 97.5 | 96.2 | 98.1 | 98.6 | not given | — |

| Comparison | 3/9 | 3/8 | 3/6 | 1/5 |
|---|---|---|---|---|
| DQuLearn vs QuClassi rule, qubit | +0.21 pts; 1.002 [1.002, 1.003], differs | +0.54 pts; 1.008 [0.992, 1.023], no difference | -0.05 pts; 0.999 [0.997, 1.002], no difference | -0.23 pts; 0.998 [0.993, 1.003], no difference |
| DQuLearn qubit vs logistic regression | -0.61 pts; 0.994 [0.986, 1.001], differs | -2.67 pts; 0.965 [0.912, 1.017], no difference | +0.73 pts; 1.008 [1.002, 1.013], differs | -1.33 pts; 0.987 [0.914, 1.059], no difference |
| DQuLearn vs QuClassi rule, photonic | -4.06 pts; 0.966 [0.203, 1.730], no difference | +5.11 pts; 1.107 [0.381, 1.833], no difference | +7.40 pts; 1.168 [0.034, 2.302], no difference | +26.30 pts; 1.509 [0.493, 2.526], no difference |
| Photonic vs qubit, DQuLearn rule | -12.33 pts; 0.869 [0.304, 1.435], no difference | -6.62 pts; 0.908 [0.516, 1.299], no difference | -6.67 pts; 0.930 [0.631, 1.230], no difference | -3.75 pts; 0.960 [0.784, 1.137], no difference |
| Photonic DQuLearn vs logistic regression | -12.94 pts; 0.864 [0.298, 1.431], no difference | -9.29 pts; 0.878 [0.458, 1.297], no difference | -5.95 pts; 0.937 [0.640, 1.235], no difference | -5.08 pts; 0.948 [0.732, 1.164], no difference |
| Exact shift vs two-point, photonic | -1.75 pts; 0.970 [0.843, 1.096], no difference | -0.10 pts; 0.998 [0.987, 1.009], no difference | +1.78 pts; 1.023 [0.925, 1.122], no difference | -6.89 pts; 0.917 [0.553, 1.281], no difference |

Parity within 2 points on every pair: qubit model True, photonic model False. Largest parameter difference between local, 1-worker and 2-worker training (3/6, seed 0): qubit 0, photonic 0.

## Epoch time with more workers (claim worker-scaling)

One-epoch wall-clock (s), mean ± std over 3 repetitions, paper circuit counts, 10,000 shots, per-circuit compilation. Scope: partial (local substitute environment).

| Cell (qubits/execution/workers/layers) | Seconds |
|---|---:|
| 5q/batched/w1/l1 | 7.36 ± 0.16 |
| 5q/batched/w1/l2 | 15.05 ± 0.59 |
| 5q/batched/w1/l3 | 20.56 ± 0.63 |
| 5q/manager_cached/w1/l1 | 3.30 ± 0.04 |
| 5q/manager_cached/w1/l2 | 10.10 ± 0.23 |
| 5q/manager_cached/w1/l3 | 15.47 ± 0.26 |
| 5q/manager_cached/w2/l1 | 1.91 ± 0.11 |
| 5q/manager_cached/w2/l2 | 5.50 ± 0.14 |
| 5q/manager_cached/w2/l3 | 8.67 ± 0.62 |
| 5q/manager_cached/w4/l1 | 1.26 ± 0.13 |
| 5q/manager_cached/w4/l2 | 3.25 ± 0.21 |
| 5q/manager_cached/w4/l3 | 4.86 ± 0.37 |
| 5q/manager_per_circuit/w1/l1 | 39.67 ± 0.40 |
| 5q/manager_per_circuit/w1/l2 | 84.55 ± 1.35 |
| 5q/manager_per_circuit/w1/l3 | 129.23 ± 2.21 |
| 5q/manager_per_circuit/w2/l1 | 22.74 ± 1.87 |
| 5q/manager_per_circuit/w2/l2 | 43.78 ± 2.17 |
| 5q/manager_per_circuit/w2/l3 | 68.07 ± 5.24 |
| 5q/manager_per_circuit/w4/l1 | 12.73 ± 1.92 |
| 5q/manager_per_circuit/w4/l2 | 24.65 ± 0.27 |
| 5q/manager_per_circuit/w4/l3 | 39.52 ± 4.06 |
| 5q/pool/w1/l1 | 40.26 ± 2.05 |
| 5q/pool/w1/l2 | 82.10 ± 0.61 |
| 5q/pool/w1/l3 | 124.81 ± 1.62 |
| 5q/pool/w2/l1 | 21.57 ± 1.39 |
| 5q/pool/w2/l2 | 46.69 ± 2.25 |
| 5q/pool/w2/l3 | 66.08 ± 1.45 |
| 5q/pool/w4/l1 | 11.06 ± 0.50 |
| 5q/pool/w4/l2 | 23.77 ± 0.51 |
| 5q/pool/w4/l3 | 40.06 ± 4.01 |
| 7q/batched/w1/l1 | 8.77 ± 0.11 |
| 7q/batched/w1/l2 | 19.53 ± 0.63 |
| 7q/batched/w1/l3 | 27.31 ± 0.36 |
| 7q/manager_per_circuit/w1/l1 | 55.95 ± 0.66 |
| 7q/manager_per_circuit/w1/l2 | 117.54 ± 0.82 |
| 7q/manager_per_circuit/w1/l3 | 179.24 ± 1.02 |
| 7q/manager_per_circuit/w2/l1 | 28.94 ± 0.41 |
| 7q/manager_per_circuit/w2/l2 | 60.94 ± 0.46 |
| 7q/manager_per_circuit/w2/l3 | 93.89 ± 0.95 |
| 7q/manager_per_circuit/w4/l1 | 16.04 ± 0.60 |
| 7q/manager_per_circuit/w4/l2 | 34.58 ± 0.38 |
| 7q/manager_per_circuit/w4/l3 | 54.53 ± 0.65 |

| Width and layers | 4 vs 1 worker (ratio) | Paper, 4 vs 1 worker (ratio, Google Cloud) | Manager vs process pool, 4 workers | Batched, 1 process, vs manager with 4 workers |
|---|---|---:|---|---|
| 5q/l1 | 0.321 [0.207, 0.434], differs | 0.729 | 1.157 [0.592, 1.723], no difference | 0.585 [0.410, 0.759], differs |
| 5q/l2 | 0.292 [0.280, 0.303], differs | 0.627 | 1.037 [0.997, 1.078], differs | 0.611 [0.549, 0.672], differs |
| 5q/l3 | 0.306 [0.220, 0.392], differs | 0.568 | 0.998 [0.536, 1.460], no difference | 0.523 [0.432, 0.613], differs |
| 7q/l1 | 0.287 [0.269, 0.305], differs | not reported | not run | 0.547 [0.510, 0.585], differs |
| 7q/l2 | 0.294 [0.282, 0.306], differs | not reported | not run | 0.565 [0.516, 0.613], differs |
| 7q/l3 | 0.304 [0.295, 0.313], differs | not reported | not run | 0.501 [0.482, 0.520], differs |

## Completion time under multi-tenancy (claim multi-tenant-gain)

Completion time from the common start (s), mean ± std over seeds 0, 1, 2; workers of 5, 10, 15, 20 qubits, one simulation slot each. Scope: partial (local substitute environment).

| Run (sweep/candidate) | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan |
|---|---:|---:|---:|---:|---:|
| multi_tenant/single_least_cpu | 78.1 ± 1.5 | 89.6 ± 2.3 | 73.6 ± 1.7 | 85.4 ± 1.9 | 89.6 ± 2.3 |
| multi_tenant/sequential_least_cpu | 49.9 ± 32.9 | 86.6 ± 30.2 | 59.2 ± 31.4 | 103.9 ± 50.6 | 128.6 ± 9.2 |
| multi_tenant/multi_round_robin | 40.5 ± 4.7 | 59.4 ± 6.7 | 93.3 ± 5.6 | 101.5 ± 6.1 | 101.5 ± 6.1 |
| multi_tenant/multi_least_cpu | 49.3 ± 8.5 | 69.3 ± 10.2 | 98.5 ± 5.1 | 106.9 ± 5.5 | 106.9 ± 5.5 |
| multi_tenant_variants_rerun/multi_fig6_sizes | 40.6 ± 0.6 | 58.4 ± 0.8 | 71.7 ± 1.2 | 77.7 ± 1.8 | 77.7 ± 1.8 |
| multi_tenant_variants_rerun/single_least_cpu | 75.8 ± 5.0 | 85.7 ± 5.4 | 71.8 ± 4.4 | 81.9 ± 5.2 | 85.7 ± 5.4 |
| multi_tenant_variants_rerun/multi_least_cpu | 42.3 ± 5.9 | 59.2 ± 6.2 | 84.6 ± 3.5 | 92.4 ± 3.9 | 92.4 ± 3.9 |
| multi_tenant_variants_rerun/multi_four_slots | 66.5 ± 2.0 | 96.7 ± 1.6 | 115.1 ± 2.3 | 124.2 ± 2.3 | 124.2 ± 2.3 |
| multi_tenant_variants_rerun/single_fig6_sizes | 62.9 ± 0.1 | 68.8 ± 0.1 | 49.6 ± 0.3 | 57.5 ± 0.2 | 68.8 ± 0.1 |

| Comparison (paired ratio) | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan |
|---|---|---|---|---|---|
| multi vs single tenant (multi_tenant) | 0.630 [0.385, 0.876], differs | 0.771 [0.533, 1.009], differs | 1.338 [1.228, 1.447], differs | 1.251 [1.159, 1.344], differs | 1.192 [1.107, 1.277], differs |
| multi vs jobs one after another (multi_tenant) | 1.393 [-1.321, 4.106], no difference | 0.872 [-0.036, 1.781], no difference | 2.160 [-1.378, 5.698], no difference | 1.324 [-0.943, 3.590], no difference | 0.834 [0.673, 0.994], differs |
| least-CPU vs round-robin, multi-tenant (multi_tenant) | 1.225 [0.587, 1.862], no difference | 1.173 [0.657, 1.690], no difference | 1.057 [0.909, 1.206], no difference | 1.054 [0.899, 1.209], no difference | 1.054 [0.899, 1.209], no difference |
| multi vs single tenant (rerun, another day) | 0.560 [0.355, 0.764], differs | 0.692 [0.513, 0.872], differs | 1.179 [1.092, 1.266], differs | 1.130 [1.043, 1.218], differs | 1.080 [0.998, 1.162], differs |

Paper (text): 5Q/1L -68.7%, 7Q/2L -8.2%; bars in `results/multi_tenant.md`.
