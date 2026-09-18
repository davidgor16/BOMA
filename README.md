# BOMA

**Balanced Optimization and Metric Alignment for Automatic Pronunciation Assessment**

Research code and saved experiments associated with *Beyond the Majority Bin: A Methodological Framework for Balanced Automatic Pronunciation Assessment*.

This repository preserves the six model implementations and experimental artifacts from the experiment server. Preparation changes to the research Python files are limited to English comments and docstrings; their executable tokens and syntax trees were checked against the captured originals. Independent tools provide explicit output paths, artifact verification, and evaluation of saved predictions.

> **Release preparation status:** the code and artifact packages have been prepared locally. The artifact manifest records whether download URLs are available. See [validation and limitations](docs/REPRODUCIBILITY.md) before interpreting this snapshot as a complete reproduction of every manuscript result.

## What is included

| Directory | Contents |
|---|---|
| `01_gopt/` | GOPT architecture, training scripts and upstream documentation |
| `02_HiPAMA/` | HiPAMA architecture and training scripts |
| `03_HierTFR/` | HierTFR, pretraining and fine-tuning scripts |
| `04_ConPCO/` | ConPCO and contrastive phonemic ordinal regularization |
| `05_hmamba/` | HMamba, feature loaders and MDD utilities |
| `06_M3C/` | M3C architecture and training scripts |
| `models_analysis/` | Original CCC, CCC-BS, PCC, MSE and distribution analyses |
| `tools/` | Separate launch, evaluation, restoration and verification tools |
| `environments/` | Environment recipes by model and for analysis |
| `metadata/` | Source provenance, checksums, launch settings and artifact index |

The saved experiment collection contains `original`, `CV`, `CV-ML`, `CV-MS`, `BOMA`, and `BOMA-SB` conditions for each architecture. These results are distributed separately from Git history. See [data and experiment layout](docs/DATA.md) and the [experiment catalogue](docs/EXPERIMENTS.md).

## Quick start: inspect saved predictions

Clone the repository and create a Python 3.10 environment. The following Bash example installs only analysis dependencies:

```bash
git clone https://github.com/davidgor16/BOMA.git
cd BOMA
python3.10 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r environments/analysis.txt
```

Restore the GOPT experiment package from the release assets. Before the upload is complete, place the prepared parts in a directory and supply `--from-dir /path/to/assets`:

```bash
python tools/restore_artifacts.py --package gopt-experiments --from-dir /path/to/assets
python tools/evaluate_saved.py \
  --experiment 01_gopt/exp/gopt-BOMA \
  --bins uniform5 --min-support 1 --seed 0 --iterations 1000 \
  --output runs/gopt-boma-evaluation.json
```

The output records the chosen binning, minimum support, scaling and seed. `uniform5` and `min-support 1` select the manuscript-oriented diagnostic convention; the standalone archived MSE driver uses minimum support 5, and the archived CCC-BS driver uses binary bins for stress and completeness. These differences are explicit in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). No original metric implementation is rewritten by the helper.

## Training

Follow [environment setup](docs/ENVIRONMENTS.md), then restore the model's data and any pretrained assets. Inspect the exact command before running:

```bash
python tools/run_experiment.py gopt --variant server --output runs/gopt-new --dry-run
python tools/run_experiment.py gopt --variant server --output runs/gopt-new
```

Available model names are `gopt`, `hipama`, `hiertfr`, `conpco`, `hmamba`, and `m3c`. `server` selects the captured BOMA training script; `sb` selects its score-balanced-loss counterpart. The launcher refuses existing output directories and paths inside archived data or experiment folders. It records the invocation and redirects logs to the new output directory. Training still uses the original model-specific working directory and relative input paths.

The historical shell launchers remain present for provenance. They contain fixed experiment paths and should only be used when those paths are deliberately intended. The new launcher invokes the Python entry point directly and does not run the historical summary or MDD post-processing stages.

## Small execution check

After installing `environments/gopt.txt` and restoring `gopt-data` and `gopt-experiments`:

```bash
python tools/smoke_gopt.py --device cpu --output runs/gopt-smoke.json
python tools/verify_source.py
```

The smoke check loads a real checkpoint, computes nine outputs for four real utterances, and checks the backward pass. It performs no optimizer step and saves no weights or prediction arrays. This is a bounded execution check, not a repeat of full training.

## Method and reproducibility

BOMA aligns score-distribution-aware optimization, checkpoint selection and evaluation. The captured BOMA training scripts implement DB-MSE and phoneme Macro-MSE selection. CCC-BS balances the contribution of occupied score intervals during evaluation. The nine targets span phoneme accuracy; word accuracy, stress and total; and utterance accuracy, completeness, fluency, prosody and total.

The source snapshot uses speaker-group `GroupKFold`; it does not implement iterative multilabel stratification simply by passing a multilabel matrix as `y`. Other known issues include HMamba prediction alignment and metric-bin conventions. These observations are documented without changing the research behavior. Consult [the complete reproducibility record](docs/REPRODUCIBILITY.md).

## Citation and attribution

See [CITATION.md](CITATION.md) for the current citation status and [THIRD_PARTY.md](THIRD_PARTY.md) for upstream projects and licenses. No journal acceptance, DOI, or global license is inferred for this preparation snapshot.
