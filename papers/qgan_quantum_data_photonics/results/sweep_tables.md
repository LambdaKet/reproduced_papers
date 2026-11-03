# Sweep tables

Every candidate of every sweep, generated from the stored run metrics. Common
rules:

- Metric: root fidelity |⟨g|τ⟩| with the phases the generator intends to set,
  averaged over the last 20 rounds of a run (10 for the chip replays);
  higher is better. The oracle ceiling reports the best fidelity over 20
  restarts and its average over phase-noise draws.
- Aggregation: mean ± standard deviation over the targets of a run (one target
  per seed; no data split, since each run learns one state).
- "Round mean reaches 0.9": first round at which the mean over targets reaches
  0.9. "Convergence round": our rule (first round after which the 20-round
  trailing mean stays within 0.01 of the final value); it ends near the run
  length on long runs, so it is not compared with the paper.
- Selection: none, except the calibration below. Each table states its
  coverage (complete when every planned candidate completed with finite
  metrics from one code state) and the code commit (`True`: the working tree
  also held uncommitted files when the runs started).

## calibration

Calibration grid (revised in phase 02, before any calibration run): learning rate x steps per turn x generator stop threshold; sigma = 0.01 pi, 1500 counts, calibration seeds 1000-1019, 400 rounds.

calibration: COMPLETE SWEEP, 8/8 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| lr_0.3pi_steps_10_stop_0.03 | 20 | 0.945 ± 0.101 | 0.943 | 243 | 363 ± 26 | COMPLETED |
| lr_0.3pi_steps_10_stop_0.05 | 20 | 0.929 ± 0.202 | 0.935 | 304 | 279 ± 95 | COMPLETED |
| lr_0.3pi_steps_20_stop_0.03 | 20 | 0.986 ± 0.005 | 0.983 | 80 | 268 ± 100 | COMPLETED |
| lr_0.3pi_steps_20_stop_0.05 | 20 | 0.991 ± 0.005 | 0.987 | 61 | 179 ± 103 | COMPLETED |
| lr_0.5pi_steps_10_stop_0.03 | 20 | 0.955 ± 0.015 | 0.953 | 100 | 377 ± 20 | COMPLETED |
| lr_0.5pi_steps_10_stop_0.05 | 20 | 0.978 ± 0.012 | 0.976 | 79 | 340 ± 64 | COMPLETED |
| lr_0.5pi_steps_20_stop_0.03 | 20 | 0.963 ± 0.021 | 0.968 | 31 | 367 ± 41 | COMPLETED |
| lr_0.5pi_steps_20_stop_0.05 | 20 | 0.988 ± 0.010 | 0.986 | 36 | 273 ± 97 | COMPLETED |

Selection rule: eligible: mean final fidelity >= 0.98; closest round to 59; tie within 5 rounds -> smaller steps_per_turn, smaller learning_rate_pi, larger generator_stop_d. Tied candidates: lr_0.3pi_steps_20_stop_0.05. Selected: lr_0.3pi_steps_20_stop_0.05 (mean final fidelity 0.9912, mean reaches 0.9 at round 61).

## phase_noise_sweep

Fig. 2c: sigma = 0.01 to 0.05 pi.

phase_noise_sweep: COMPLETE SWEEP, 5/5 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| sigma_0.01pi | 100 | 0.993 ± 0.004 | 0.990 | 57 | 169 ± 93 | COMPLETED |
| sigma_0.02pi | 100 | 0.984 ± 0.007 | 0.974 | 64 | 265 ± 103 | COMPLETED |
| sigma_0.03pi | 100 | 0.980 ± 0.011 | 0.955 | 63 | 321 ± 73 | COMPLETED |
| sigma_0.04pi | 100 | 0.969 ± 0.014 | 0.926 | 71 | 355 ± 43 | COMPLETED |
| sigma_0.05pi | 100 | 0.962 ± 0.018 | 0.896 | 88 | 370 ± 30 | COMPLETED |

## phase_noise_variants

Ambiguity checks of the Fig. 2c sweep: exact tr[M tau] at every sigma; one phase-noise draw per round at 0.02 and 0.04 pi.

phase_noise_variants: COMPLETE SWEEP, 7/7 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| exact_tau_sigma_0.01pi | 100 | 0.994 ± 0.003 | 0.992 | 52 | 162 ± 87 | COMPLETED |
| exact_tau_sigma_0.02pi | 100 | 0.986 ± 0.007 | 0.975 | 58 | 257 ± 108 | COMPLETED |
| exact_tau_sigma_0.03pi | 100 | 0.980 ± 0.008 | 0.957 | 63 | 321 ± 71 | COMPLETED |
| exact_tau_sigma_0.04pi | 100 | 0.975 ± 0.013 | 0.934 | 69 | 354 ± 36 | COMPLETED |
| exact_tau_sigma_0.05pi | 100 | 0.962 ± 0.017 | 0.895 | 86 | 371 ± 25 | COMPLETED |
| round_redraw_sigma_0.02pi | 100 | 0.986 ± 0.005 | 0.975 | 51 | 219 ± 98 | COMPLETED |
| round_redraw_sigma_0.04pi | 100 | 0.968 ± 0.009 | 0.927 | 76 | 353 ± 44 | COMPLETED |

## phase_noise_diagnosis

Gap diagnosis for Fig. 2c at high sigma: one training-protocol component changed at a time (generator stop threshold, step size, steps per turn) on targets 0-19.

phase_noise_diagnosis: COMPLETE SWEEP, 10/10 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| stop_0.01_sigma_0.01pi | 20 | 0.956 ± 0.019 | 0.954 | 99 | 386 ± 11 | COMPLETED |
| stop_0.01_sigma_0.03pi | 20 | 0.898 ± 0.044 | 0.876 | 136 | 393 ± 5 | COMPLETED |
| stop_0.01_sigma_0.05pi | 20 | 0.827 ± 0.069 | 0.782 | None | 396 ± 4 | COMPLETED |
| stop_0.03_sigma_0.01pi | 20 | 0.986 ± 0.006 | 0.984 | 64 | 277 ± 79 | COMPLETED |
| stop_0.03_sigma_0.03pi | 20 | 0.957 ± 0.013 | 0.936 | 80 | 375 ± 22 | COMPLETED |
| stop_0.03_sigma_0.05pi | 20 | 0.928 ± 0.044 | 0.870 | 112 | 388 ± 11 | COMPLETED |
| step_0.5pi_sigma_0.03pi | 20 | 0.967 ± 0.015 | 0.944 | 37 | 379 ± 14 | COMPLETED |
| step_0.5pi_sigma_0.05pi | 20 | 0.945 ± 0.016 | 0.879 | 49 | 387 ± 12 | COMPLETED |
| steps_10_sigma_0.03pi | 20 | 0.962 ± 0.023 | 0.938 | 174 | 324 ± 70 | COMPLETED |
| steps_10_sigma_0.05pi | 20 | 0.940 ± 0.029 | 0.877 | 251 | 377 ± 24 | COMPLETED |

## defect_rate_sweep_low

Fig. 3b, part of the defect-rate sweep: k = 0..4 stuck phase shifters per circuit (run in parallel with the other parts).

defect_rate_sweep_low: COMPLETE SWEEP, 5/5 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| stuck_0_of_30 | 100 | 0.985 ± 0.007 | 0.975 | 64 | 1796 ± 281 | COMPLETED |
| stuck_2_of_30 | 100 | 0.987 ± 0.006 | 0.976 | 66 | 1785 ± 210 | COMPLETED |
| stuck_4_of_30 | 100 | 0.988 ± 0.005 | 0.977 | 63 | 1716 ± 347 | COMPLETED |
| stuck_6_of_30 | 100 | 0.988 ± 0.006 | 0.978 | 63 | 1686 ± 356 | COMPLETED |
| stuck_8_of_30 | 100 | 0.989 ± 0.008 | 0.977 | 61 | 1519 ± 532 | COMPLETED |

## defect_rate_sweep_mid

Fig. 3b, part of the defect-rate sweep: k = 5..8 stuck phase shifters per circuit (run in parallel with the other parts).

defect_rate_sweep_mid: COMPLETE SWEEP, 4/4 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| stuck_10_of_30 | 100 | 0.982 ± 0.063 | 0.970 | 68 | 1299 ± 623 | COMPLETED |
| stuck_12_of_30 | 100 | 0.965 ± 0.104 | 0.953 | 107 | 1253 ± 650 | COMPLETED |
| stuck_14_of_30 | 100 | 0.913 ± 0.192 | 0.902 | 182 | 1385 ± 688 | COMPLETED |
| stuck_16_of_30 | 100 | 0.877 ± 0.209 | 0.870 | 1800 | 1589 ± 650 | COMPLETED |

## defect_rate_sweep_high

Fig. 3b, part of the defect-rate sweep: k = 9..12 stuck phase shifters per circuit (run in parallel with the other parts).

defect_rate_sweep_high: COMPLETE SWEEP, 4/4 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| stuck_18_of_30 | 100 | 0.810 ± 0.204 | 0.798 | None | 1829 ± 484 | COMPLETED |
| stuck_20_of_30 | 100 | 0.645 ± 0.225 | 0.644 | None | 1990 ± 21 | COMPLETED |
| stuck_22_of_30 | 100 | 0.534 ± 0.203 | 0.517 | None | 1994 ± 10 | COMPLETED |
| stuck_24_of_30 | 100 | 0.460 ± 0.203 | 0.448 | None | 1988 ± 30 | COMPLETED |

## shot_noise_sweep

Suppl. Fig. 9b: counts 100 to 2000, sigma = 0.

shot_noise_sweep: COMPLETE SWEEP, 8/8 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| counts_100 | 100 | 0.961 ± 0.016 | 0.962 | 76 | 2476 ± 20 | COMPLETED |
| counts_200 | 100 | 0.980 ± 0.009 | 0.981 | 71 | 2412 ± 83 | COMPLETED |
| counts_300 | 100 | 0.986 ± 0.006 | 0.986 | 66 | 2274 ± 271 | COMPLETED |
| counts_400 | 100 | 0.989 ± 0.005 | 0.989 | 55 | 2145 ± 437 | COMPLETED |
| counts_500 | 100 | 0.992 ± 0.004 | 0.992 | 59 | 1871 ± 677 | COMPLETED |
| counts_1000 | 100 | 0.997 ± 0.002 | 0.997 | 57 | 676 ± 637 | COMPLETED |
| counts_1500 | 100 | 0.998 ± 0.001 | 0.998 | 59 | 387 ± 385 | COMPLETED |
| counts_2000 | 100 | 0.998 ± 0.001 | 0.998 | 59 | 343 ± 357 | COMPLETED |

## single_runs

Single-condition runs: Suppl. Sec. 6 (one stuck phase shifter per circuit) and our fixed-discriminator ablation.

single_runs: COMPLETE SWEEP, 2/2 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| single_defect | 100 | 0.987 ± 0.006 | 0.976 | 66 | 1785 ± 210 | COMPLETED |
| fixed_discriminator | 100 | 0.793 ± 0.209 | 0.792 | None | 381 ± 23 | COMPLETED |

## single_defect_positions

Suppl. Sec. 6 per position: each generator phase shifter stuck in both circuits.

single_defect_positions: COMPLETE SWEEP, 15/15 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| position_0 | 20 | 0.988 ± 0.005 | 0.977 | 67 | 1720 ± 390 | COMPLETED |
| position_1 | 20 | 0.990 ± 0.006 | 0.977 | 49 | 1788 ± 198 | COMPLETED |
| position_2 | 20 | 0.988 ± 0.005 | 0.977 | 47 | 1792 ± 178 | COMPLETED |
| position_3 | 20 | 0.984 ± 0.009 | 0.975 | 52 | 1772 ± 232 | COMPLETED |
| position_4 | 20 | 0.981 ± 0.017 | 0.970 | 63 | 1830 ± 215 | COMPLETED |
| position_5 | 20 | 0.987 ± 0.007 | 0.976 | 69 | 1786 ± 233 | COMPLETED |
| position_6 | 20 | 0.987 ± 0.007 | 0.978 | 68 | 1794 ± 211 | COMPLETED |
| position_7 | 20 | 0.985 ± 0.010 | 0.975 | 48 | 1765 ± 295 | COMPLETED |
| position_8 | 20 | 0.989 ± 0.005 | 0.978 | 58 | 1778 ± 231 | COMPLETED |
| position_9 | 20 | 0.987 ± 0.007 | 0.976 | 49 | 1756 ± 285 | COMPLETED |
| position_10 | 20 | 0.988 ± 0.005 | 0.978 | 64 | 1743 ± 244 | COMPLETED |
| position_11 | 20 | 0.987 ± 0.006 | 0.976 | 59 | 1688 ± 389 | COMPLETED |
| position_12 | 20 | 0.985 ± 0.007 | 0.975 | 67 | 1838 ± 173 | COMPLETED |
| position_13 | 20 | 0.988 ± 0.006 | 0.977 | 48 | 1804 ± 185 | COMPLETED |
| position_14 | 20 | 0.988 ± 0.005 | 0.978 | 62 | 1703 ± 279 | COMPLETED |

## chip_emulations

Fig. 2a-b and Fig. 3a chip runs replayed in simulation at sigma = 0.01 and 0.02 pi.

chip_emulations: COMPLETE SWEEP, 4/4 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| intact_chip_emulation_sigma_0.01pi | 20 | 0.989 ± 0.004 | 0.987 | 55 | 88 ± 30 | COMPLETED |
| intact_chip_emulation_sigma_0.02pi | 20 | 0.982 ± 0.009 | 0.971 | 63 | 101 ± 24 | COMPLETED |
| defective_chip_emulation_sigma_0.01pi | 20 | 0.940 ± 0.091 | 0.942 | 118 | 135 ± 28 | COMPLETED |
| defective_chip_emulation_sigma_0.02pi | 20 | 0.902 ± 0.127 | 0.898 | 135 | 138 ± 25 | COMPLETED |

## oracle_ceiling

Our control: best reachable fidelity for the defect-sweep targets and stuck sets.

oracle_ceiling: COMPLETE SWEEP, 14/14 runs; code [('c9a60a4df29eb4be9a1249eaee836666a6c086ee', True)]

| Candidate | Targets | Best fidelity (mean ± std) | Noise-averaged σ = 0.0π | Noise-averaged σ = 0.01π | Noise-averaged σ = 0.02π | Noise-averaged σ = 0.03π | Noise-averaged σ = 0.04π | Noise-averaged σ = 0.05π | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ceiling_stuck_0_of_30 | 100 | 1.000 ± 0.000 | 1.000 | 0.997 | 0.989 | 0.975 | 0.957 | 0.933 | COMPLETED |
| ceiling_stuck_2_of_30 | 100 | 1.000 ± 0.001 | 1.000 | 0.997 | 0.989 | 0.975 | 0.956 | 0.933 | COMPLETED |
| ceiling_stuck_4_of_30 | 100 | 1.000 ± 0.001 | 1.000 | 0.997 | 0.989 | 0.975 | 0.956 | 0.933 | COMPLETED |
| ceiling_stuck_6_of_30 | 100 | 1.000 ± 0.001 | 1.000 | 0.997 | 0.989 | 0.975 | 0.956 | 0.933 | COMPLETED |
| ceiling_stuck_8_of_30 | 100 | 1.000 ± 0.002 | 1.000 | 0.997 | 0.989 | 0.975 | 0.956 | 0.933 | COMPLETED |
| ceiling_stuck_10_of_30 | 100 | 0.998 ± 0.015 | 0.998 | 0.995 | 0.987 | 0.973 | 0.954 | 0.931 | COMPLETED |
| ceiling_stuck_12_of_30 | 100 | 0.994 ± 0.028 | 0.994 | 0.991 | 0.983 | 0.969 | 0.950 | 0.927 | COMPLETED |
| ceiling_stuck_14_of_30 | 100 | 0.986 ± 0.040 | 0.986 | 0.984 | 0.975 | 0.962 | 0.943 | 0.920 | COMPLETED |
| ceiling_stuck_16_of_30 | 100 | 0.975 ± 0.049 | 0.975 | 0.973 | 0.965 | 0.951 | 0.933 | 0.910 | COMPLETED |
| ceiling_stuck_18_of_30 | 100 | 0.953 ± 0.057 | 0.953 | 0.950 | 0.942 | 0.929 | 0.911 | 0.889 | COMPLETED |
| ceiling_stuck_20_of_30 | 100 | 0.893 ± 0.078 | 0.893 | 0.890 | 0.883 | 0.871 | 0.854 | 0.833 | COMPLETED |
| ceiling_stuck_22_of_30 | 100 | 0.820 ± 0.087 | 0.820 | 0.818 | 0.811 | 0.800 | 0.785 | 0.765 | COMPLETED |
| ceiling_stuck_24_of_30 | 100 | 0.711 ± 0.101 | 0.711 | 0.709 | 0.703 | 0.694 | 0.680 | 0.664 | COMPLETED |
| ceiling_chip_defect_set | 20 | 0.998 ± 0.004 | 0.998 | 0.995 | 0.987 | 0.974 | 0.955 | 0.931 | COMPLETED |

## noise_free_reference

Phase 05: noise-free QGAN at the run lengths of Fig. 2c (400 rounds) and Suppl. Fig. 9b (2500 rounds).

noise_free_reference: COMPLETE SWEEP, 2/2 runs; code [('3b8d6a3ca08ab9513e51a9664eaba8e0f5fe6146', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| noise_free_400_rounds | 100 | 0.990 ± 0.003 | 0.990 | 53 | 89 ± 31 | COMPLETED |
| noise_free_2500_rounds | 100 | 0.991 ± 0.002 | 0.991 | 53 | 96 ± 65 | COMPLETED |

## merlin_equivalence

Phase 04: MerLin chip with the phase-03 noise model, against the phase-03 numpy runs on the same 100 targets.

merlin_equivalence: COMPLETE SWEEP, 3/3 runs; code [('b6ed959628b1ae4218dc5613e4e433d3f665fed5', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| merlin_injected_sigma_0.01pi | 100 | 0.993 ± 0.004 | 0.990 | 57 | 169 ± 93 | COMPLETED |
| merlin_injected_sigma_0.05pi | 100 | 0.962 ± 0.018 | 0.896 | 88 | 370 ± 30 | COMPLETED |
| merlin_injected_stuck_16_of_30 | 100 | 0.877 ± 0.209 | 0.870 | 1800 | 1589 ± 650 | COMPLETED |

## merlin_native_phase_noise

Phase 04: Fig. 2c with MerLin's phase_error noise model (Uniform, same std as the paper's sigma).

merlin_native_phase_noise: COMPLETE SWEEP, 5/5 runs; code [('b6ed959628b1ae4218dc5613e4e433d3f665fed5', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| merlin_phase_error_sigma_0.01pi | 100 | 0.992 ± 0.005 | 0.989 | 49 | 179 ± 102 | COMPLETED |
| merlin_phase_error_sigma_0.02pi | 100 | 0.985 ± 0.007 | 0.973 | 60 | 272 ± 100 | COMPLETED |
| merlin_phase_error_sigma_0.03pi | 100 | 0.978 ± 0.010 | 0.954 | 62 | 332 ± 65 | COMPLETED |
| merlin_phase_error_sigma_0.04pi | 100 | 0.966 ± 0.017 | 0.923 | 76 | 366 ± 34 | COMPLETED |
| merlin_phase_error_sigma_0.05pi | 100 | 0.947 ± 0.026 | 0.885 | 88 | 378 ± 21 | COMPLETED |

## merlin_native_defects

Phase 04: Fig. 3b at 5 predeclared rates with MerLin's phase_error noise model, sigma = 0.02 pi.

merlin_native_defects: COMPLETE SWEEP, 5/5 runs; code [('b6ed959628b1ae4218dc5613e4e433d3f665fed5', True)]

| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | Round mean reaches 0.9 | Convergence round (mean ± std) | Status |
|---|---:|---:|---:|---:|---:|---|
| merlin_phase_error_stuck_0_of_30 | 100 | 0.984 ± 0.009 | 0.973 | 60 | 1845 ± 175 | COMPLETED |
| merlin_phase_error_stuck_8_of_30 | 100 | 0.988 ± 0.012 | 0.976 | 67 | 1520 ± 450 | COMPLETED |
| merlin_phase_error_stuck_14_of_30 | 100 | 0.915 ± 0.191 | 0.900 | 227 | 1407 ± 652 | COMPLETED |
| merlin_phase_error_stuck_16_of_30 | 100 | 0.897 ± 0.177 | 0.882 | 1401 | 1595 ± 558 | COMPLETED |
| merlin_phase_error_stuck_20_of_30 | 100 | 0.623 ± 0.226 | 0.619 | None | 1993 ± 15 | COMPLETED |
