# BOMA fold distribution similarity

`distribution_similarity.py` compares the full canonical training partition with
the training and validation subsets of every fold in the **BOMA experiment of
each of the six models**. It does not include the baseline, CV ablations or BOMA-SB.

## Percentage definition

For normalized frequency vectors `p` and `q`, the overlap percentage is:

```text
overlap(p, q) = 100 * sum(min(p[i], q[i]))
              = 100 * (1 - 0.5 * sum(abs(p[i] - q[i])))
```

100% means identical distributions on the chosen support; 0% means disjoint
support. This is histogram intersection (one minus total variation), not a
probability of equivalence, statistical significance or a stratification guarantee.
It is not a percentage conversion of Jensen-Shannon or Wasserstein distance.

- **Phonemes:** count each of the 39 scored canonical phoneme identities at most
  once per utterance, then normalize the resulting count vector to sum to one.
  Repeated tokens within an utterance do not increase its presence count.
  Padding and HMamba's unscored silence/boundary entries are excluded.
- **UTT-Mean:** arithmetic mean of the five utterance annotations (accuracy,
  completeness, fluency, prosody and total), on their original 0–10 scale.
  This is distinct from U-TOT. By default, use 20 equal-width bins (0.5 points):
  `[0,0.5), ..., [9.5,10]`. Scores are computed in float64 and rounded to six
  decimal places before binning. This makes boundary placement reproducible;
  historical plots computed after float32 normalization can place exact boundary
  values in adjacent bins. The percentage depends on binning, so always report it.

Each subset is compared with its model's **full 2,500-utterance training partition**,
not directly with the other subset. The summary averages 30 comparisons for
training and 30 for validation, giving equal weight to models and folds. Some
models reuse the same partitions; these are descriptive comparisons, not 30
independent replications. Higher overlap for a fold's training subset is also
expected because it contains 80% of the reference partition.

## Reproduce the calculation

Use Python 3.10 or newer, NumPy and SciPy. The repository's analysis environment
can be installed with:

```bash
python -m pip install -r environments/analysis.txt
```

Restore the training annotations from the release assets. Restoring `gopt-data`
and `hmamba-data` is sufficient: the remaining models' required label files have
the same SHA-256 hashes as the corresponding GOPT files, as recorded in
`metadata/artifact_files.json`. The helper permits this substitution only for
byte-identical files, and reports the source artifact for every model.

```bash
gh release download v1.0.0 --repo davidgor16/BOMA \
  --pattern 'gopt-data.tar.gz.part*' \
  --pattern 'hmamba-data.tar.gz.part*' --dir release-assets
python tools/restore_artifacts.py --package gopt-data --package hmamba-data \
  --from-dir release-assets
python model_utils/distribution_similarity.py --output-dir runs/distribution-similarity
```

If the artifacts are already restored elsewhere, pass their common parent:

```bash
python model_utils/distribution_similarity.py \
  --data-root /path/to/restored-artifacts \
  --output-dir runs/distribution-similarity
```

`--data-root` may be repeated. No training, downloading or artifact modification
is performed by the analysis script. Only `summary.json` and `fold_similarity.csv`
are written to the requested output directory; existing files with these names
are replaced. Missing or mismatched labels and changed utterance ordering fail
explicitly. `--utt-bins N` changes the UTT-Mean binning and records the resulting
edges in the output.

## Results with 20 UTT-Mean bins

| Distribution | Subset vs full training | Mean overlap | Minimum | Maximum |
|---|---|---:|---:|---:|
| Phonemes | Fold training | 99.5234% | 99.3929% | 99.6076% |
| Phonemes | Fold validation | 98.0923% | 97.6089% | 98.4323% |
| UTT-Mean | Fold training | 97.5053% | 96.8100% | 98.7500% |
| UTT-Mean | Fold validation | 90.0213% | 87.2400% | 95.0000% |

[Per-model and per-fold CSV](results/fold_similarity.csv) and
[full JSON report](results/summary.json) include all 60 comparisons. The JSON
also includes input hashes, frequency counts, explicit bin edges, per-model
summaries, Jensen-Shannon distance for phonemes and Wasserstein-1 distance for
UTT-Mean in original score points. The latter compares the full empirical score
distributions without histogram bins.

These values describe broadly similar phonetic composition and approximate
preservation of the score distribution. They do not establish identical tails,
equal representation of all score regions, or automatic joint stratification.

## Fold provenance and archived figures

The script reads explicit validation-speaker lists from
[boma_fold_assignments.json](boma_fold_assignments.json). It never calls a splitter.
This avoids changes in tie ordering between software environments.

**These assignments are reconstructed and checked against archived distribution
figures; they are not independently saved historical membership logs.** The
archive contains two sets of fold distributions. The regenerated server profile
matches four models; an alternative GroupKFold profile matches the other two.
The assignment file identifies each model's profile and records SHA-256 references
for all 132 archived distribution PNGs. The existing `metadata/group_folds.json`
describes a later regeneration of the current procedure, and should not be treated
as evidence that every historical experiment used that same assignment. It is
preserved unchanged.

For the actual distribution plots, restore the corresponding `*-experiments`
package from [release v1.0.0](https://github.com/davidgor16/BOMA/releases/tag/v1.0.0).
Each experiment contains `global_train_*_dist.png` and each `fold_N/` contains
the corresponding training and validation PNGs:

| Model | BOMA artifact directory |
|---|---|
| GOPT | `01_gopt/exp/gopt-BOMA/` |
| HiPAMA | `02_HiPAMA/exp/hipama-BOMA/` |
| HierTFR | `03_HierTFR/exp/hiertfr-BOMA/` |
| ConPCO | `04_ConPCO/exp/conpco-BOMA/` |
| HMamba | `05_hmamba/exp/hmamba-BOMA/0/` |
| M3C | `06_M3C/exp/M3C-BOMA/` |

The historical red plots are titled "Utterance Total Score Distribution", but
their source computes the mean of all five utterance annotations: UTT-Mean.
