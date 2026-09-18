# Data, checkpoints and results

All distributed artifacts come from the captured experiment-server tree. They retain their original paths and bytes. The source preparation does not change arrays, labels, checkpoint weights, tables, logs or figures.

Large files are packaged separately from Git history. [artifact_packages.json](../metadata/artifact_packages.json) lists archive parts and their checksums; [artifact_files.json](../metadata/artifact_files.json) records every extracted file's size and SHA-256. Packages are split into parts no larger than 900 MiB. GitHub supports binary assets through [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Restore selected packages

Each model has `<model>-data`, `<model>-experiments`, and `<model>-supplementary` packages where applicable. `experiments` includes saved predictions, checkpoints, CSVs and experiment figures; `supplementary` includes logs and auxiliary outputs. HierTFR's pretraining checkpoint is included in `hiertfr-experiments`.

```bash
python tools/restore_artifacts.py \
  --package gopt-data --package gopt-experiments \
  --from-dir /path/to/downloaded-release-assets
```

Once published URLs are populated in the manifest, `--from-dir` can be omitted to download the selected parts into `.artifact-cache/`. The tool verifies each part and every extracted file. It refuses to overwrite different content and rejects paths outside the chosen destination. Existing identical files are verified and reused.

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
