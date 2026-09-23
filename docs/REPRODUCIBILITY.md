# Reproducibility record

## Source and permitted edits

The publication source is the experiment-server tree captured on 2026-09-18. No file from the earlier local research directory was selected as a substitute. The captured BOMA scripts use DB-MSE and phoneme Macro-MSE model selection.

English annotation edits are checked against the captured files in two ways: equality of executable Python tokens, excluding comments and recognized docstrings, and equality of the AST after removing docstrings. Numerical operations, arguments, seeds, split logic, model topology and saved outputs are retained. A subsequent terminology-only update renames Macro-Loss experiment paths to Dynamic Balanced MSE (DB); the source manifest records these path changes separately from the original annotation-only checks. [source_manifest.json](../metadata/source_manifest.json) records original and publication hashes. The new `tools/` scripts are separate conveniences and are not edits to the research algorithms.

## Validation scope

- The SSH transfer verified all 2,687 captured files against SHA-256 hashes computed while reading the server originals.
- All 73 research Python files passed annotation-preservation checks.
- GOPT loaded a real BOMA checkpoint strictly and produced finite outputs and gradients for four real utterances, without optimizer updates or checkpoint/prediction writes. CPU execution passed on Windows/Python 3.11/PyTorch 2.5.1 and in the observed Linux/Python 3.10/PyTorch 2.1.1 environment.
- All six original training entry points displayed CLI help successfully in the server environment. This is an import/argument-parsing check, not six successful training runs.
- Saved-prediction evaluation and archive-restoration checks are recorded in [validation.json](../metadata/validation.json).
- All 2,418 files in the 18 artifact packages passed decompression and per-file hash verification. A fresh GOPT data/experiment restoration passed for 390 files. Evaluation of its five restored BOMA folds with 1,000 bootstrap iterations, seed 0 and five uniform bins reproduced all nine CCC-BS means in manuscript Table VII to three decimal places. This verifies that table's GOPT means, not all architectures or all reported statistics.

Full retraining of all architectures and clean installation of all six environments have not been validated by this preparation task.

### Saved figure title correction (v1.0.1)

The [figure title audit](FIGURE_TITLES.md) covers 330 UTT-Mean histograms across all
30 experiment directories containing these plots. Every pixel outside the title
rectangle is identical after correction. All other members of the six rebuilt
experiment packages retain their original SHA-256 hashes. The correction also
passed a check of all 694 images in the original testGPU inventory: only the 330
authorized files changed. Research training scripts and numerical artifacts are
unchanged.

## Protocol distinctions requiring attention

### Speaker partitioning

The current scripts use `GroupKFold(n_splits=5)`. This separates speaker groups but does not perform iterative multilabel stratification. Passing the multilabel matrix as `y` does not change this: [GroupKFold ignores `y`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html). The manuscript and supplementary material now describe speaker-grouped cross-validation to match this implementation; phonetic and proficiency labels do not stratify the folds.

[group_folds.json](../metadata/group_folds.json) records the fold indices regenerated with the server's scikit-learn 1.7.2, original utterance-order files and the exact current GroupKFold procedure. Each model has 2,500 utterances and 125 speakers; each validation fold contains 500 utterances and 25 speakers, with no speaker overlap with training. These are regenerated current-procedure assignments, not independent proof of the historical fold membership used for every archived run.

The [BOMA distribution audit](../models_analysis/README.md) identifies two fold profiles
in the archived distribution figures. It uses explicit reconstructed speaker
assignments matching those profiles, independently of the installed splitter,
and calculates distribution overlap percentages for all six BOMA experiments.
The assignment provenance, figure hashes and distinction from the later
`group_folds.json` regeneration are documented with the script and results.

### Metric conventions

The standalone Macro-MSE script uses `min_support=5`; the manuscript describes including every occupied bin (`min_support=1`). CCC-BS uses right-closed intervals, whereas the original MSE function uses the left-closed default of `np.digitize`. The helper makes the support and bin choices explicit and records them without changing these original functions.

The archived CCC-BS driver uses binary intervals for word stress and utterance completeness. A GOPT-BOMA diagnostic gives U-COM approximately 0.07520 with binary intervals and 0.04625 with five uniform intervals; the latter agrees with the manuscript's rounded 0.046. Saved U-COM labels have more than two distinct values. The version of labels and bin convention used for each manuscript table should be identified before claiming exact reproduction.

Bootstrap resampling gives equal influence to included bins; it does not create additional independent observations. Both sample and population standard deviations across model runs are reported separately by the helper. These are distinct from variability across bootstrap draws.

### HMamba alignment

For the five `hmamba-BOMA` and five `hmamba-original` runs, phoneme predictions have sequence length 52 while the corresponding targets have length 50; word arrays may contain 15,966 predictions and 15,967 targets. Two baseline arrays are missing entirely. The archived heatmap implementation contains its own truncation logic, which is preserved as part of that source. The new generic evaluator instead rejects incompatible shapes because array length alone cannot establish semantic alignment.

### DB-MSE constant-target branch

When all targets in a batch are equal, the preserved loss returns the expected numerical MSE from inside a `torch.no_grad()` block. The returned value consequently has no gradient. This edge case is recorded without changing its behavior.

### Historical experiment variants

Saved conditions include `original`, `CV`, `CV-DB`, `CV-MS`, `BOMA`, and `BOMA-SB`. The Dynamic Balanced MSE condition now uses `CV-DB`. Previously released archives retain their historical member names; restoration maps these to DB without changing file contents or checksums. Separate immutable training-source versions for every historical condition are not all available in this tree. The new launcher exposes only the captured BOMA and SB scripts; it does not invent missing ablation implementations.

### Launchers and repeatability

Historical launchers contain fixed output directories. HierTFR additionally refers to an absent `collect_summary_bc.py`; ConPCO's launcher folder label and selected Python script do not agree. The new helper invokes the selected Python script directly and writes into a fresh output directory. This does not repair historical launchers or change the selected model algorithm.

Several archived scripts do not set all random seeds, and GPU kernels can introduce nondeterminism. A current package snapshot and a successful smoke check do not establish bit-for-bit historical retraining reproducibility.
