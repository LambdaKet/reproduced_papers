# Runtime gaps

Scope: partial: local substitute environment (worker processes on one laptop instead of IBM-Q or Google Cloud VMs), paper circuit counts, 3 repetitions. Amdahl fit T_w / T_1 = s + (1 − s) / w on the 4-worker time; s is the share of the one-worker epoch that does not run in parallel.

| Layers | Source | 4 workers / 1 | Not parallel, s | Not parallel, s per circuit | Parallel, s per circuit | 2 workers / 1, measured | 2 workers / 1, fit |
|---|---|---|---|---|---|---|---|
| 1 | paper | 0.729 | 0.64 | 0.1654 | 0.0936 | 0.899 | 0.819 |
| 1 | reproduction | 0.321 | 0.09 | 0.0026 | 0.0250 | 0.573 | 0.547 |
| 2 | paper | 0.627 | 0.50 | 0.1674 | 0.1656 | 0.915 | 0.751 |
| 2 | reproduction | 0.292 | 0.06 | 0.0016 | 0.0277 | 0.518 | 0.528 |
| 3 | paper | 0.568 | 0.42 | 0.1706 | 0.2317 | 0.811 | 0.712 |
| 3 | reproduction | 0.306 | 0.07 | 0.0022 | 0.0277 | 0.526 | 0.537 |

Seconds per circuit per worker (single-tenant: four jobs assumed to run together; paper from the Fig. 5 and Fig. 6 bars):

| Setting | Paper | This reproduction | Ratio |
|---|---|---|---|
| One worker, 5Q/1L epoch | 0.259 | 0.0275 | ×9 |
| Single-tenant, four jobs on four workers | 0.895 | 0.0319 | ×28 |

Multi / single ratio paired by seed (95% interval), workers without latency and with 0.3 s per circuit (jobs at one tenth of Fig. 6's sizes), and the paper:

| Workers | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan | Circuits per second, 5Q/1L |
|---|---|---|---|---|---|---|
| Compute only | 0.750 [-0.075, 1.575] | 0.954 [-0.039, 1.947] | 1.469 [0.305, 2.633] | 1.329 [0.339, 2.319] | 1.134 [0.228, 2.040] | 1.514 [-0.052, 3.081] |
| 0.3 s latency per circuit | 0.263 [0.185, 0.341] | 0.390 [0.320, 0.461] | 0.732 [0.653, 0.812] | 0.678 [0.618, 0.738] | 0.578 [0.527, 0.629] | 3.841 [2.770, 4.913] |
| Paper, text | 0.313 | — | — | 0.918 | — | 3.9 |
| Paper, Fig. 6 bars | 0.250 | 1.000 | 0.984 | 0.818 | 0.818 (longest job) | 4.05 |

Changes: compute only 5Q/1L −25.0%, makespan +13.4%; with latency 5Q/1L −73.7%, makespan −42.2%.

Per-circuit milliseconds of the DQuLearn manager, 5 qubits, 1–3 layers (range of the nine runs per worker count); communication is the manager's round trip minus the worker's execution:

| Workers | Communication | Execution |
|---|---|---|
| 1 | 0.37–0.41 | 26.8–30.0 |
| 2 | 0.36–0.46 | 28.9–33.8 |
| 4 | 0.45–0.63 | 30.8–38.7 |

7-qubit circuits assigned to the 5-qubit worker: 0 over the 27 runs of `multi_tenant` and `multi_tenant_variants_rerun`, every tenancy mode included.
