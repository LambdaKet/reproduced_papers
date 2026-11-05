# Worker scaling

Scope: partial: local substitute environment (worker processes on one laptop instead of IBM-Q or Google Cloud VMs), paper circuit counts, 3 repetitions. Epoch wall-clock in seconds, mean ± std over 3 repetitions; reductions are 1 − (paired ratio to 1 worker), with 95% intervals.

| Qubits | Family | Layers | 1 worker | 2 workers | 4 workers | Reduction, 2 workers | Reduction, 4 workers | Paper, 4 workers |
|---|---|---|---|---|---|---|---|---|
| 5 | manager_per_circuit | 1 | 39.67 ± 0.40 | 22.74 ± 1.87 | 12.73 ± 1.92 | 42.7% [31.7, 53.6] | 67.9% [56.6, 79.3] | 27.1% |
| 5 | manager_per_circuit | 2 | 84.55 ± 1.35 | 43.78 ± 2.17 | 24.65 ± 0.27 | 48.2% [41.2, 55.2] | 70.8% [69.7, 72.0] | 37.3% |
| 5 | manager_per_circuit | 3 | 129.23 ± 2.21 | 68.07 ± 5.24 | 39.52 ± 4.06 | 47.4% [39.6, 55.1] | 69.4% [60.8, 78.0] | 43.2% |
| 5 | manager_cached | 1 | 3.30 ± 0.04 | 1.91 ± 0.11 | 1.26 ± 0.13 | 42.1% [32.0, 52.2] | 61.6% [50.7, 72.6] | — |
| 5 | manager_cached | 2 | 10.10 ± 0.23 | 5.50 ± 0.14 | 3.25 ± 0.21 | 45.5% [41.8, 49.2] | 67.8% [61.8, 73.8] | — |
| 5 | manager_cached | 3 | 15.47 ± 0.26 | 8.67 ± 0.62 | 4.86 ± 0.37 | 44.0% [35.7, 52.3] | 68.6% [62.6, 74.6] | — |
| 5 | pool | 1 | 40.26 ± 2.05 | 21.57 ± 1.39 | 11.06 ± 0.50 | 46.4% [42.4, 50.5] | 72.4% [66.2, 78.7] | — |
| 5 | pool | 2 | 82.10 ± 0.61 | 46.69 ± 2.25 | 23.77 ± 0.51 | 43.1% [36.7, 49.5] | 71.0% [70.0, 72.1] | — |
| 5 | pool | 3 | 124.81 ± 1.62 | 66.08 ± 1.45 | 40.06 ± 4.01 | 47.1% [44.2, 49.9] | 67.9% [61.0, 74.8] | — |
| 7 | manager_per_circuit | 1 | 55.95 ± 0.66 | 28.94 ± 0.41 | 16.04 ± 0.60 | 48.3% [47.8, 48.7] | 71.3% [69.5, 73.1] | — |
| 7 | manager_per_circuit | 2 | 117.54 ± 0.82 | 60.94 ± 0.46 | 34.58 ± 0.38 | 48.1% [46.3, 50.0] | 70.6% [69.4, 71.8] | — |
| 7 | manager_per_circuit | 3 | 179.24 ± 1.02 | 93.89 ± 0.95 | 54.53 ± 0.65 | 47.6% [46.9, 48.3] | 69.6% [68.7, 70.5] | — |

Batched execution in one process, and the manager against a plain process pool:

| Qubits | Layers | Batched, one process (s) | Ratio batched / manager with 4 workers | Ratio manager / pool, 4 workers |
|---|---|---|---|---|
| 5 | 1 | 7.36 ± 0.16 | 0.585 [0.410, 0.759] | 1.157 [0.592, 1.723] |
| 5 | 2 | 15.05 ± 0.59 | 0.611 [0.549, 0.672] | 1.037 [0.997, 1.078] |
| 5 | 3 | 20.56 ± 0.63 | 0.523 [0.432, 0.613] | 0.998 [0.536, 1.460] |
| 7 | 1 | 8.77 ± 0.11 | 0.547 [0.510, 0.585] | — |
| 7 | 2 | 19.53 ± 0.63 | 0.565 [0.516, 0.613] | — |
| 7 | 3 | 27.31 ± 0.36 | 0.501 [0.482, 0.520] | — |
