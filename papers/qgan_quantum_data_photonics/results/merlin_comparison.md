### Equivalence: MerLin chip with the phase-03 noise draws (100 targets each)

Final root fidelity per target, MerLin against numpy.

| Condition | Targets identical (within 1e-6) | Mean difference | Largest per-round difference | Holds |
|---|---:|---:|---:|---|
| σ = 0.01π | 100% | 0.0e+00 | 0.0e+00 | yes |
| σ = 0.05π | 100% | 0.0e+00 | 0.0e+00 | yes |
| 16 / 30 stuck, σ = 0.02π | 100% | 0.0e+00 | 0.0e+00 | yes |

### MerLin phase noise, Fig. 2c conditions (1500 counts)

Final root fidelity, mean ± std over the same 100 targets.

| Phase noise σ | Numpy chip | MerLin chip | MerLin − numpy [95 % CI] | Paper (read) |
|---|---:|---:|---:|---:|
| 0.01π | 0.993 ± 0.004 | 0.992 ± 0.005 | -0.001 [-0.002, +0.000] | 1.000 |
| 0.02π | 0.984 ± 0.007 | 0.985 ± 0.007 | +0.000 [-0.002, +0.002] | 0.990 |
| 0.03π | 0.980 ± 0.011 | 0.978 ± 0.010 | -0.003 [-0.006, +0.000] | 0.970 |
| 0.04π | 0.969 ± 0.014 | 0.966 ± 0.017 | -0.003 [-0.007, +0.002] | 0.920 |
| 0.05π | 0.962 ± 0.018 | 0.947 ± 0.026 | -0.015 [-0.021, -0.009] | 0.780 |

Claim rule (mean ≥ 0.90 for σ ≤ 0.03π): numpy holds, MerLin holds. Means within 0.02: 5 of 5 conditions.

### MerLin phase noise, Fig. 3b conditions (σ = 0.02π, 1500 counts)

Final root fidelity, mean ± std over the same 100 targets.

| Stuck phase shifters | Numpy chip | MerLin chip | MerLin − numpy [95 % CI] | Paper (read) |
|---|---:|---:|---:|---:|
| 0 / 30 | 0.985 ± 0.007 | 0.984 ± 0.009 | -0.001 [-0.004, +0.001] | 0.985 |
| 8 / 30 | 0.989 ± 0.008 | 0.988 ± 0.012 | -0.001 [-0.003, +0.001] | 0.970 |
| 14 / 30 | 0.913 ± 0.192 | 0.915 ± 0.191 | +0.001 [-0.013, +0.017] | 0.913 |
| 16 / 30 | 0.877 ± 0.209 | 0.897 ± 0.177 | +0.021 [+0.001, +0.042] | 0.867 |
| 20 / 30 | 0.645 ± 0.225 | 0.623 ± 0.226 | -0.022 [-0.058, +0.013] | 0.709 |

Claim rule (mean ≥ 0.90 up to 14/30 stuck): numpy holds, MerLin holds. Means within 0.02: 3 of 5 conditions.
