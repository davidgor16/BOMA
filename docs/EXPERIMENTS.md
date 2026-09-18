# Saved experiment catalogue

Paths and condition names are preserved from the experiment server. A prediction set normally contains phoneme, word and utterance predictions and targets. The counts below do not certify semantic alignment or completeness; see REPRODUCIBILITY.md for HMamba exceptions.

| Model | Condition | Runs/folds found | Files | Size (MB) |
|---|---|---:|---:|---:|
| 01_gopt | `gopt-BOMA` | 5 | 63 | 10.87 |
| 01_gopt | `gopt-BOMA-SB` | 5 | 63 | 10.80 |
| 01_gopt | `gopt-CV` | 5 | 63 | 10.87 |
| 01_gopt | `gopt-CV-ML` | 5 | 63 | 10.87 |
| 01_gopt | `gopt-CV-MS` | 5 | 63 | 10.87 |
| 01_gopt | `gopt-original` | 5 | 41 | 8.49 |
| 02_HiPAMA | `hipama-BOMA` | 5 | 63 | 10.96 |
| 02_HiPAMA | `hipama-BOMA-SB` | 5 | 63 | 10.89 |
| 02_HiPAMA | `hipama-CV` | 5 | 63 | 10.96 |
| 02_HiPAMA | `hipama-CV-ML` | 5 | 63 | 10.96 |
| 02_HiPAMA | `hipama-CV-MS` | 5 | 63 | 10.96 |
| 02_HiPAMA | `hipama-original` | 5 | 41 | 8.58 |
| 03_HierTFR | `hiertfr-BOMA` | 5 | 63 | 16.49 |
| 03_HierTFR | `hiertfr-BOMA-SB` | 5 | 63 | 16.49 |
| 03_HierTFR | `hiertfr-CV` | 5 | 63 | 16.49 |
| 03_HierTFR | `hiertfr-CV-ML` | 5 | 63 | 16.49 |
| 03_HierTFR | `hiertfr-CV-MS` | 5 | 63 | 16.49 |
| 03_HierTFR | `hiertfr-original` | 5 | 40 | 14.10 |
| 04_ConPCO | `conpco-BOMA` | 5 | 63 | 20.17 |
| 04_ConPCO | `conpco-BOMA-SB` | 5 | 63 | 20.10 |
| 04_ConPCO | `conpco-CV` | 5 | 63 | 20.17 |
| 04_ConPCO | `conpco-CV-ML` | 5 | 63 | 20.17 |
| 04_ConPCO | `conpco-CV-MS` | 5 | 63 | 20.17 |
| 04_ConPCO | `conpco-original` | 5 | 40 | 17.78 |
| 05_hmamba | `hmamba-BOMA` | 5 | 62 | 60.28 |
| 05_hmamba | `hmamba-BOMA-SB` | 5 | 62 | 60.38 |
| 05_hmamba | `hmamba-CV` | 5 | 62 | 60.38 |
| 05_hmamba | `hmamba-CV-ML` | 5 | 62 | 60.38 |
| 05_hmamba | `hmamba-CV-MS` | 5 | 62 | 60.38 |
| 05_hmamba | `hmamba-original` | 5 | 80 | 70.12 |
| 06_M3C | `M3C-BOMA` | 5 | 73 | 17.12 |
| 06_M3C | `M3C-BOMA-SB` | 5 | 73 | 17.12 |
| 06_M3C | `M3C-CV` | 5 | 73 | 17.19 |
| 06_M3C | `M3C-CV-ML` | 5 | 73 | 17.19 |
| 06_M3C | `M3C-CV-MS` | 5 | 73 | 17.19 |
| 06_M3C | `M3C-original` | 5 | 51 | 14.81 |

Exact paths and sizes are recorded in [experiments.json](../metadata/experiments.json). Baselines use architecture-specific run folders. CV-based conditions use `fold_1` through `fold_5`; HMamba has an intermediate `0/` directory.
