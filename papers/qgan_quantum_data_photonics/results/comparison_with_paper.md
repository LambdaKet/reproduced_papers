### Phase noise (Fig. 2c): final root fidelity, 100 targets, 1500 counts

| σ | Ours (mean ± std) | Paper (read) | Ours − paper | Round to 0.9 (ours / paper) | Exact tr[Mτ] | Ours, noise-averaged | Ceiling, noise-averaged |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01π | 0.993 ± 0.004 | 1.00 | -0.007 | 57 / 59 | 0.994 | 0.990 | 0.997 |
| 0.02π | 0.984 ± 0.007 | 0.99 | -0.006 | 64 / 79 | 0.986 | 0.974 | 0.989 |
| 0.03π | 0.980 ± 0.011 | 0.97 | +0.010 | 63 / 159 | 0.980 | 0.955 | 0.975 |
| 0.04π | 0.969 ± 0.014 | 0.92 | +0.049 | 71 / 245 | 0.975 | 0.926 | 0.957 |
| 0.05π | 0.962 ± 0.018 | 0.78 | +0.182 | 88 / None | 0.962 | 0.896 | 0.933 |

Rules: {'every level within 0.03 of the paper': False, 'mean falls monotonically with sigma': True, 'mean crosses 0.90 between 0.03 pi and 0.05 pi': False, 'mean below 0.90 at some sigma <= 0.03 pi': False}. Verdict: **partially supported**.

### Defect rate (Fig. 3b): final root fidelity, 100 targets, σ = 0.02π, 1500 counts

| Stuck / 30 | Ours (mean ± std) | Targets ≥ 0.9 | Paper (read) | Ours − paper | Ceiling | QGAN − ceiling [95 % CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.985 ± 0.007 | 100% | 0.985 | +0.000 | 1.000 | -0.015 [-0.016, -0.014] |
| 2 | 0.987 ± 0.006 | 100% | 0.979 | +0.008 | 1.000 | -0.013 [-0.014, -0.012] |
| 4 | 0.988 ± 0.005 | 100% | 0.980 | +0.008 | 1.000 | -0.012 [-0.014, -0.011] |
| 6 | 0.988 ± 0.006 | 100% | 0.969 | +0.019 | 1.000 | -0.012 [-0.013, -0.011] |
| 8 | 0.989 ± 0.008 | 100% | 0.970 | +0.019 | 1.000 | -0.011 [-0.012, -0.010] |
| 10 | 0.982 ± 0.063 | 98% | 0.958 | +0.024 | 0.998 | -0.016 [-0.027, -0.009] |
| 12 | 0.965 ± 0.104 | 94% | 0.944 | +0.021 | 0.994 | -0.028 [-0.046, -0.015] |
| 14 | 0.913 ± 0.192 | 85% | 0.913 | +0.000 | 0.986 | -0.073 [-0.105, -0.043] |
| 16 | 0.877 ± 0.209 | 79% | 0.867 | +0.010 | 0.975 | -0.099 [-0.133, -0.067] |
| 18 | 0.810 ± 0.204 | 50% | 0.799 | +0.011 | 0.953 | -0.143 [-0.176, -0.114] |
| 20 | 0.645 ± 0.225 | 13% | 0.709 | -0.064 | 0.893 | -0.248 [-0.284, -0.212] |
| 22 | 0.534 ± 0.203 | 4% | 0.648 | -0.114 | 0.820 | -0.286 [-0.319, -0.254] |
| 24 | 0.460 ± 0.203 | 0% | 0.456 | +0.004 | 0.711 | -0.251 [-0.281, -0.222] |

Rules: {'mean >= 0.90 up to 14/30 stuck': True, 'mean < 0.90 at 10/30 or fewer': False, 'mean decreases with the defect rate (steps of at most +0.01)': True, 'QGAN within 0.02 of the ceiling at every rate': False}. Verdict: **supported**; mechanism: learning-limited at high defect rates (QGAN below the ceiling).

### Shot noise (Suppl. Fig. 9): final root fidelity, 100 targets, σ = 0

| Counts | Ours (mean ± std) | Paper (read) |
|---:|---:|---:|
| 100 | 0.961 ± 0.016 | 0.921 |
| 200 | 0.980 ± 0.009 | 0.987 |
| 300 | 0.986 ± 0.006 | 0.990 |
| 400 | 0.989 ± 0.005 | 0.989 |
| 500 | 0.992 ± 0.004 | 0.989 |
| 1000 | 0.997 ± 0.002 | 0.986 |
| 1500 | 0.998 ± 0.001 | 0.989 |
| 2000 | 0.998 ± 0.001 | 0.990 |

TVD of a uniform 16-outcome distribution: 0.1551 at 100 counts (paper ≈0.155), 0.0155 at 10,000 (paper ≈0.016).

### Chip runs replayed in simulation, single defect, fixed discriminator

- intact chip, σ = 0.01π: 0.989 ± 0.004 (20 targets; paper chip run 0.9602)
- intact chip, σ = 0.02π: 0.982 ± 0.009 (20 targets; paper chip run 0.9602)
- defective chip, σ = 0.01π: 0.940 ± 0.091 (20 targets; paper chip run 0.9211)
- defective chip, σ = 0.02π: 0.902 ± 0.127 (20 targets; paper chip run 0.9211)
- Rule for the intact chip: {'mean >= 0.90 within 150 rounds at sigma 0.01 pi or 0.02 pi': True}.
- One stuck phase shifter per circuit: 0.987 ± 0.006 (paper 0.9833 ± 0.0700; no defect 0.9914 ± 0.0123).
- Fixed discriminator minus QGAN at σ = 0.02π (paired, 100 targets): -0.191 [-0.232, -0.152].
