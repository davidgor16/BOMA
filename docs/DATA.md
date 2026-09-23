# Data, checkpoints and results

All distributed artifacts come from the captured experiment-server tree. Arrays, labels, checkpoint weights, tables and logs retain their original bytes. Release v1.0.1 makes only the documented UTT-Mean histogram title corrections; all pixels outside the title rectangles are unchanged. See the [figure integrity audit](FIGURE_TITLES.md).

Large files are packaged separately from Git history: the six corrected experiment packages are in [v1.0.1](https://github.com/davidgor16/BOMA/releases/tag/v1.0.1), while data and supplementary packages remain in [v1.0.0](https://github.com/davidgor16/BOMA/releases/tag/v1.0.0). There are 18 packages in 22 archive parts. [artifact_packages.json](../metadata/artifact_packages.json) lists the current links and checksums; [artifact_files.json](../metadata/artifact_files.json) records every extracted file's size and SHA-256. Packages are split into parts no larger than 900 MiB. All 660 experiment PNGs are included: 330 have corrected titles, and the remaining 330 retain their exact original bytes.

## Restore selected packages

Each model has `<model>-data`, `<model>-experiments`, and `<model>-supplementary` packages where applicable. `experiments` includes saved predictions, checkpoints, CSVs and experiment figures; `supplementary` includes logs and auxiliary outputs. HierTFR's pretraining checkpoint is included in `hiertfr-experiments`.

While the repository is private, use the release page from an authorized GitHub account, or install and authenticate the [GitHub CLI](https://cli.github.com/manual/gh_auth_login). For example, download both GOPT packages with [gh release download](https://cli.github.com/manual/gh_release_download):

```bash
gh auth login
gh release download v1.0.0 --repo davidgor16/BOMA \
  --pattern 'gopt-data.tar.gz.part*' --dir release-assets
gh release download v1.0.1 --repo davidgor16/BOMA \
  --pattern 'gopt-experiments.tar.gz.part*' --dir release-assets
```

Keep every numbered part of each selected package in the same directory. Restore them from the repository root:

```bash
python tools/restore_artifacts.py \
  --package gopt-data --package gopt-experiments \
  --from-dir release-assets
```

The tool verifies each part and every extracted file. It refuses to overwrite different content and rejects paths outside the chosen destination. Existing identical files are verified and reused. Restore to a fresh destination if older experiment PNGs are present. If the repository becomes public, `--from-dir` can be omitted to download the published URLs directly into `.artifact-cache/`; the helper itself does not authenticate to private GitHub repositories.

## Input layout

| Model | Input location |
|---|---|
| GOPT / HiPAMA | `data/seq_data_librispeech/` |
| HierTFR | `data/seq_data_librispeech_v3/` |
| ConPCO | `data/seq_data_librispeech_v4/` |
| HMamba | `data/so762/` with GOP, three SSL feature sets and raw-audio features |
| M3C | `data/GoP_VC_Features_Embbeding_norm/` |

Paths in this table are relative to each model directory. Phoneme dictionaries, utterance ordering files and model YAML configuration are retained alongside the source. Do not reorder samples or infer cross-model alignment solely from array length.

The original data README files retain upstream acquisition instructions. Speechocean762 and derived features must retain their original attribution and applicable dataset terms; the source-code licenses do not replace the dataset terms.

## Known completeness issues

Two expected HMamba baseline arrays are absent in the server snapshot: `05_hmamba/exp/hmamba-original/1/preds/utt_pred.npy` and `05_hmamba/exp/hmamba-original/2/preds/word_target.npy`. Ten HMamba prediction sets require alignment review. These issues are preserved and reported; no fabricated replacements are supplied.

Operating-system metadata (`__MACOSX` and Finder metadata) is excluded from publication. Environments, Python bytecode and existing Git internals are also excluded. The untouched transfer snapshot is retained separately during preparation.
