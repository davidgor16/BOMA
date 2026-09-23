# UTT-Mean title correction

Release [v1.0.1](https://github.com/davidgor16/BOMA/releases/tag/v1.0.1) replaces
the misleading title `Utterance Total Score Distribution` with
`UTT-Mean Distribution` in **330 saved histograms**. These span 30 experiment
directories: CV, CV-DB, CV-MS, BOMA and BOMA-SB for all six architectures.
Each architecture contributes 55 images (five global, 25 training-fold and
25 validation-fold histograms). Baseline directories without these histograms
require no correction.

The histogram filenames remain `*_utt_score_dist.png`, preserving paths and
links. The graphs describe the mean of the five utterance annotations, as
computed by the experiment scripts. Genuine U-TOT plots, including
`models_analysis/histogram_U-TOT.png` on testGPU, are unchanged.

## Integrity checks

The correction uses Pillow to replace only the title rectangle, without
regenerating the chart or using generative image editing. All originals are
2,100 by 1,200 RGBA PNGs. Their dimensions, mode and DPI are preserved.

After saving and reopening each corrected image, every RGBA pixel outside the
authorized rectangle is compared with the original. **All 330 comparisons have
zero changed pixels outside the title.** Bars, numerical tick labels, legends,
percentile lines and plot frames are therefore pixel-identical. The report also
records the changed-pixel count inside the title and both file hashes.

See [the per-image audit](../metadata/figure_title_corrections.json) and
[the correction script](../models_analysis/correct_distribution_titles.py).
The audit uses half-open rectangle coordinates `(x0, y0, x1, y1)`. It records
server paths and publication paths separately where the existing ML-to-DB
terminology mapping applies.

The six replacement experiment packages were rebuilt from the verified v1.0.0
archives. Every non-target member was checked against its original SHA-256:
arrays, labels, checkpoints, tables, logs and the remaining 330 experiment PNGs
retain their exact bytes. GitHub verifies the uploaded assets' SHA-256 hashes.
The original release remains available as an immutable reference; no data or
supplementary packages were republished.

## Download and restore

Use the current [package manifest](../metadata/artifact_packages.json) to locate
each package. Experiment packages come from v1.0.1; data and supplementary
packages continue to come from v1.0.0. For example:

```bash
gh release download v1.0.1 --repo davidgor16/BOMA \
  --pattern 'gopt-experiments.tar.gz.part*' --dir corrected-release-assets
python tools/restore_artifacts.py --package gopt-experiments \
  --from-dir corrected-release-assets --destination corrected-artifacts
```

Use a fresh destination when an older restoration contains the original PNGs:
the restoration tool deliberately refuses to overwrite different bytes.

To apply the same audited edit to a separate copy of the original figures:

```bash
python models_analysis/correct_distribution_titles.py \
  --input-root original-figures --output-root corrected-figures \
  --report title-audit.json
```

The script accepts the inspected original layout only, stops on unexpected
dimensions or title geometry, and writes to a separate output tree. It leaves
training-source files unchanged.
