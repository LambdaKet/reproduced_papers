# Multi-tenant

Scope: partial: local substitute environment (worker processes on one laptop instead of IBM-Q or Google Cloud VMs), paper circuit counts, 3 repetitions. Completion time of each job from the common start (s), mean ± std over 3 seeds (the seed shuffles the submission order, identically in every mode).

| Mode | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan |
|---|---|---|---|---|---|
| multi, lowest CPU (DQuLearn) | 49.3 ± 8.5 | 69.3 ± 10.2 | 98.5 ± 5.1 | 106.9 ± 5.5 | 106.9 ± 5.5 |
| single, one circuit per worker | 78.1 ± 1.5 | 89.6 ± 2.3 | 73.6 ± 1.7 | 85.4 ± 1.9 | 89.6 ± 2.3 |
| jobs one after another | 49.9 ± 32.9 | 86.6 ± 30.2 | 59.2 ± 31.4 | 103.9 ± 50.6 | 128.6 ± 9.2 |
| multi, round-robin | 40.5 ± 4.7 | 59.4 ± 6.7 | 93.3 ± 5.6 | 101.5 ± 6.1 | 101.5 ± 6.1 |

Multi- vs single-tenant, ratio paired by seed (95% interval) and the paper:

| Job | Completion, multi / single | Change | Paper, text | Paper, bars | Circuits per second, multi / single |
|---|---|---|---|---|---|
| 5Q/1L | 0.630 [0.385, 0.876] | −37.0% | −68.7% | −75.0% | 1.62 [0.94, 2.29] |
| 5Q/2L | 0.771 [0.533, 1.009] | −22.9% | — | +0.0% | 1.31 [0.88, 1.74] |
| 7Q/1L | 1.338 [1.228, 1.447] | +33.8% | — | −1.6% | 0.75 [0.69, 0.81] |
| 7Q/2L | 1.251 [1.159, 1.344] | +25.1% | −8.2% | −18.2% | 0.80 [0.74, 0.86] |

Makespan, multi / single: 1.192 [1.107, 1.277] (+19.2%). Paper 5Q/1L circuits per second: ×3.9 (text), ×4.05 (bars).

Two ambiguities, both readings (sweep `multi_tenant_variants_rerun`): multi / single ratio paired by seed (95% interval):

| Reading | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan | Circuits per second, 5Q/1L |
|---|---|---|---|---|---|---|
| One slot per worker, 7-qubit jobs as the text (rerun of the table above) | 0.560 [0.355, 0.764] | 0.692 [0.513, 0.872] | 1.179 [1.092, 1.266] | 1.130 [1.043, 1.218] | 1.080 [0.998, 1.162] | 1.812 [1.150, 2.474] |
| Up to four circuits at once per worker | 0.880 [0.784, 0.976] | 1.131 [0.994, 1.267] | 1.606 [1.416, 1.797] | 1.520 [1.328, 1.713] | 1.453 [1.271, 1.636] | 1.138 [1.011, 1.265] |
| 7-qubit jobs sized as Fig. 6 (1,440 × layers circuits) | 0.646 [0.621, 0.671] | 0.849 [0.817, 0.880] | 1.444 [1.407, 1.482] | 1.352 [1.287, 1.417] | 1.129 [1.066, 1.192] | 1.549 [1.488, 1.609] |

Four slots vs one slot, both multi-tenant: 5Q/1L 1.596 [0.933, 2.259], 5Q/2L 1.645 [1.170, 2.119], 7Q/1L 1.362 [1.280, 1.443], 7Q/2L 1.345 [1.258, 1.432], makespan 1.345 [1.258, 1.432].
