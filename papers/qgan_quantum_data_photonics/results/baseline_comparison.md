Phase-05 comparison: the QGAN against its matched baselines. Simulation, 100 targets per run (seeds 0-99), paired by target. Decision rule, set before these comparisons: tied when the mean paired difference is at most 0.02; clearly higher or lower when it exceeds 0.02 and its 95 % interval excludes 0; otherwise no clear separation. Paper values are read from its figures.

### Systems

| System | Role | Provenance | Runs (successful / expected) | Resources | Wall-clock per run (min) | Status |
|---|---|---|---:|---|---|---|
| QGAN, our reproduction | reproduced method (numpy chip simulator) | local runs, phase 03 | 27 / 27 | 30 generator + 12 discriminator phases; parameter shift; 1500 counts per setting (Suppl. Fig. 9b: 100-2000) | 400 rounds: 2.1–2.5; 2000 rounds: 6.9–15.5; 2500 rounds: 6.7–13.3 (up to 10 runs at once) | reference |
| Noise-free QGAN | baseline: cost of noise (phase-01 plan) | local runs, phase 05 | 2 / 2 | same phases and settings; sigma = 0, exact probabilities | 400 rounds: 0.7–0.7; 2500 rounds: 4.2–4.2 (1 run at a time) | COMMENSURABLE |
| Oracle ceiling | control: best fidelity the chip can reach (knows the target) | local runs, phase 03 | 13 / 13 | 30 generator phases minus stuck ones; exact gradients; 20 restarts x 2000 Adam iterations; 1000 noise draws per sigma | optimization: 0.5–0.6 (up to 10 runs at once) | COMMENSURABLE (upper bound by design) |
| Fixed discriminator | ablation: random projectors redrawn every round | local run, phase 03 | 1 / 1 | 30 generator phases; no discriminator training | 400 rounds: 0.3–0.3 (up to 10 runs at once) | COMMENSURABLE |
| MerLin adaptation, MerLin noise | photonic adaptation (MerLin QuantumLayer, 8 modes, 2 photons) | local runs, phase 04 (simulation) | 10 / 10 | same as the QGAN; MerLin phase_error noise (uniform, same standard deviation) | 400 rounds: 4.0–4.1; 2000 rounds: 18.0–28.6 (3 runs at once) | PARTIALLY_COMMENSURABLE (noise distribution; target prepared on the chip) |
| Paper QGAN | method as published | paper-reported (Figs. 2c, 3b, Suppl. Fig. 9b, Suppl. Sec. 6) | — | 1500 counts; training settings not stated | — | values read from figures |

Wall-clock: one run = 100 targets; laptop CPU (Apple M5, 10 cores), one thread per run. Runs that shared the machine are slower, so times compare workloads only roughly. MerLin with the phase-03 noise draws gives the numpy results exactly (`merlin_comparison.md`).

### Phase noise (Fig. 2c): 1500 counts, 400 rounds

| σ | QGAN | Paper | QGAN − noise-free [95 % CI] | QGAN − ceiling, noise-averaged [95 % CI] | MerLin noise − QGAN [95 % CI] |
|---|---:|---:|---|---|---|
| 0.01π | 0.993 ± 0.004 | 1.00 | +0.002 [+0.001, +0.003], tied | -0.007 [-0.009, -0.006], tied | -0.001 [-0.002, +0.000], tied |
| 0.02π | 0.984 ± 0.007 | 0.99 | -0.006 [-0.007, -0.004], tied | -0.015 [-0.017, -0.013], tied | +0.000 [-0.002, +0.002], tied |
| 0.03π | 0.980 ± 0.011 | 0.97 | -0.010 [-0.012, -0.008], tied | -0.021 [-0.023, -0.018], clearly lower | -0.003 [-0.006, +0.000], tied |
| 0.04π | 0.969 ± 0.014 | 0.92 | -0.021 [-0.024, -0.019], clearly lower | -0.030 [-0.033, -0.027], clearly lower | -0.003 [-0.007, +0.002], tied |
| 0.05π | 0.962 ± 0.018 | 0.78 | -0.028 [-0.032, -0.025], clearly lower | -0.037 [-0.040, -0.033], clearly lower | -0.015 [-0.021, -0.009], tied |

Noise-free QGAN, 400 rounds: 0.990 ± 0.003. Noise-averaged means, QGAN / ceiling: 0.01π 0.990 / 0.997; 0.02π 0.974 / 0.989; 0.03π 0.955 / 0.975; 0.04π 0.926 / 0.957; 0.05π 0.896 / 0.933.

### Stuck phase shifters (Fig. 3b): σ = 0.02π, 1500 counts, 2000 rounds

| Stuck / 30 | QGAN | Paper | Ceiling | QGAN − ceiling [95 % CI] | MerLin noise − ceiling [95 % CI] |
|---:|---:|---:|---:|---|---|
| 0 | 0.985 ± 0.007 | 0.985 | 1.000 ± 0.000 | -0.015 [-0.016, -0.014], tied | -0.016 [-0.018, -0.015], tied |
| 2 | 0.987 ± 0.006 | 0.979 | 1.000 ± 0.001 | -0.013 [-0.014, -0.012], tied | not run |
| 4 | 0.988 ± 0.005 | 0.980 | 1.000 ± 0.001 | -0.012 [-0.014, -0.011], tied | not run |
| 6 | 0.988 ± 0.006 | 0.969 | 1.000 ± 0.001 | -0.012 [-0.013, -0.011], tied | not run |
| 8 | 0.989 ± 0.008 | 0.970 | 1.000 ± 0.002 | -0.011 [-0.012, -0.010], tied | -0.012 [-0.014, -0.010], tied |
| 10 | 0.982 ± 0.063 | 0.958 | 0.998 ± 0.015 | -0.016 [-0.027, -0.009], tied | not run |
| 12 | 0.965 ± 0.104 | 0.944 | 0.994 ± 0.028 | -0.028 [-0.046, -0.015], clearly lower | not run |
| 14 | 0.913 ± 0.192 | 0.913 | 0.986 ± 0.040 | -0.073 [-0.105, -0.043], clearly lower | -0.072 [-0.104, -0.043], clearly lower |
| 16 | 0.877 ± 0.209 | 0.867 | 0.975 ± 0.049 | -0.099 [-0.133, -0.067], clearly lower | -0.078 [-0.107, -0.052], clearly lower |
| 18 | 0.810 ± 0.204 | 0.799 | 0.953 ± 0.057 | -0.143 [-0.176, -0.114], clearly lower | not run |
| 20 | 0.645 ± 0.225 | 0.709 | 0.893 ± 0.078 | -0.248 [-0.284, -0.212], clearly lower | -0.270 [-0.307, -0.235], clearly lower |
| 22 | 0.534 ± 0.203 | 0.648 | 0.820 ± 0.087 | -0.286 [-0.319, -0.254], clearly lower | not run |
| 24 | 0.460 ± 0.203 | 0.456 | 0.711 ± 0.101 | -0.251 [-0.281, -0.222], clearly lower | not run |

### Shot noise (Suppl. Fig. 9b): σ = 0, 2500 rounds

| Counts | QGAN | Paper | QGAN − noise-free [95 % CI] |
|---:|---:|---:|---|
| 100 | 0.961 ± 0.016 | 0.921 | -0.030 [-0.034, -0.027], clearly lower |
| 200 | 0.980 ± 0.009 | 0.987 | -0.011 [-0.013, -0.009], tied |
| 300 | 0.986 ± 0.006 | 0.990 | -0.004 [-0.006, -0.003], tied |
| 400 | 0.989 ± 0.005 | 0.989 | -0.002 [-0.003, -0.001], tied |
| 500 | 0.992 ± 0.004 | 0.989 | +0.001 [+0.000, +0.002], tied |
| 1000 | 0.997 ± 0.002 | 0.986 | +0.006 [+0.006, +0.007], tied |
| 1500 | 0.998 ± 0.001 | 0.989 | +0.007 [+0.007, +0.008], tied |
| 2000 | 0.998 ± 0.001 | 0.990 | +0.007 [+0.007, +0.008], tied |

Noise-free QGAN, 2500 rounds: 0.991 ± 0.002.

### One stuck phase shifter (Suppl. Sec. 6) and the discriminator

- One stuck phase shifter per circuit against none (σ = 0.02π, 2000 rounds): 0.987 ± 0.006 against 0.985 ± 0.007; difference +0.002 [+0.000, +0.004], tied. Paper: 0.9833 ± 0.0700 against 0.9914 ± 0.0123.
- Fixed discriminator against the QGAN (σ = 0.02π, intact, 400 rounds): 0.793 ± 0.209 against 0.984 ± 0.007; difference -0.191 [-0.232, -0.152], clearly lower.
