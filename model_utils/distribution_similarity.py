"""Compare the six archived BOMA experiments' folds with full training data.

Similarity is histogram overlap: 100 * sum(min(p, q)), for normalized
distributions p and q. Frozen, figure-checked speaker assignments are used;
the installed scikit-learn/NumPy versions never allocate new folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = Path(__file__).with_name("boma_fold_assignments.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_percent(p: np.ndarray, q: np.ndarray) -> float:
    """Histogram intersection, equivalent to 100 * (1 - total variation)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.ndim != 1 or p.shape != q.shape:
        raise ValueError("Distributions must have the same one-dimensional support.")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Distribution counts must be finite.")
    if (p < 0).any() or (q < 0).any() or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("Distribution counts must be nonnegative and nonempty.")
    return float(100 * np.minimum(p / p.sum(), q / q.sum()).sum())


class ArtifactReader:
    """Read verified arrays, permitting only byte-identical artifact aliases."""

    def __init__(self, roots: list[Path], manifest: dict):
        self.roots = roots
        self.manifest = manifest
        self.cache: dict[str, tuple[np.ndarray, str]] = {}

    def load(self, relative: str) -> tuple[np.ndarray, dict]:
        expected = self.manifest[relative]
        digest = expected["sha256"]
        if digest not in self.cache:
            aliases = [relative] + [
                name for name, record in self.manifest.items()
                if name != relative and record["sha256"] == digest
            ]
            for name in aliases:
                for root in self.roots:
                    path = root / name
                    # Some local artifact trees contain zero-byte placeholders.
                    if not path.is_file() or path.stat().st_size == 0:
                        continue
                    if path.stat().st_size != expected["bytes"] or sha256(path) != digest:
                        raise ValueError(f"Artifact checksum mismatch: {path}")
                    self.cache[digest] = (np.load(path, allow_pickle=False), name)
                    break
                if digest in self.cache:
                    break
            else:
                raise FileNotFoundError(
                    f"Missing verified labels: {relative}. Restore the data packages "
                    "described in model_utils/README.md, or supply --data-root."
                )
        array, source = self.cache[digest]
        return array, {
            "requested_artifact": relative,
            "read_from_artifact": source,
            "sha256": digest,
        }


def utterance_mean(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("Expected five utterance-level annotations per utterance.")
    if not np.isfinite(labels).all() or (labels < 0).any() or (labels > 10).any():
        raise ValueError("Utterance annotations must be complete and in [0, 10].")
    # Use the original annotation scale, before float32 training normalization.
    # This fixes the side of exact histogram boundaries across environments.
    return np.round(np.mean(labels.astype(np.float64), axis=1), decimals=6)


def phoneme_presence(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 3 or labels.shape[-1] not in (2, 4):
        raise ValueError("Unexpected phoneme label layout.")
    ids = labels[:, :, 0]
    # Excludes padding and unscored silence/boundary tokens in HMamba.
    valid = (ids >= 0) & (labels[:, :, -1] >= 0)
    phone_ids = np.unique(ids[valid])
    if len(phone_ids) != 39 or not np.equal(phone_ids, np.floor(phone_ids)).all():
        raise ValueError("Expected the 39 scored canonical phoneme identities.")
    presence = np.stack([((ids == pid) & valid).any(axis=1) for pid in phone_ids], axis=1)
    return presence, phone_ids.astype(int)


def fold_indices(groups: np.ndarray, validation_speakers: list[list[str]]):
    if len(validation_speakers) != 5:
        raise ValueError("Expected five folds.")
    all_validation = [speaker for fold in validation_speakers for speaker in fold]
    if len(all_validation) != len(set(all_validation)) or set(all_validation) != set(groups):
        raise ValueError("Validation folds must partition the speaker set exactly once.")
    for number, speakers in enumerate(validation_speakers, start=1):
        mask = np.isin(groups, speakers)
        train, validation = np.flatnonzero(~mask), np.flatnonzero(mask)
        if len(validation) != 500 or len(train) != 2000 or len(speakers) != 25:
            raise ValueError(f"Unexpected size for fold {number}.")
        if len(set(groups[train])) != 100 or set(groups[train]) & set(groups[validation]):
            raise ValueError(f"Speaker leakage in fold {number}.")
        yield number, train, validation


METRICS = (
    "phoneme_overlap_percent", "utt_mean_overlap_percent",
    "phoneme_jensen_shannon_distance", "utt_mean_wasserstein_points",
    "utt_mean", "utt_mean_absolute_shift",
)


def summarize(rows: list[dict]) -> dict:
    return {
        "comparisons": len(rows),
        **{
            name: {
                "mean": float(np.mean([row[name] for row in rows])),
                "minimum": float(min(row[name] for row in rows)),
                "maximum": float(max(row[name] for row in rows)),
            }
            for name in METRICS
        },
    }


def calculate(data_roots: list[Path], utt_bins: int = 20) -> tuple[list[dict], dict]:
    if utt_bins < 1:
        raise ValueError("--utt-bins must be a positive integer.")
    assignments = json.loads(ASSIGNMENTS.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "metadata/artifact_files.json").read_text(encoding="utf-8"))["files"]
    reader = ArtifactReader(data_roots, manifest)
    edges = np.linspace(0, 10, utt_bins + 1)
    rows, inputs = [], {}
    for model, spec in assignments["models"].items():
        utt_labels, utt_source = reader.load(spec["utterance_labels"])
        phn_labels, phn_source = reader.load(spec["phoneme_labels"])
        order_path = ROOT / spec["utterance_order"]
        if sha256(order_path) != spec["utterance_order_sha256"]:
            raise ValueError(f"Utterance order checksum mismatch: {model}")
        groups = np.array([
            line.split("\t")[0][:5]
            for line in order_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ])
        if len(utt_labels) != 2500 or len(phn_labels) != 2500 or len(groups) != 2500:
            raise ValueError(f"Unexpected training data size: {model}")
        scores = utterance_mean(utt_labels)
        presence, phone_ids = phoneme_presence(phn_labels)
        global_phone = presence.sum(axis=0)
        global_score = np.histogram(scores, bins=edges)[0]
        profile = assignments["profiles"][spec["fold_profile"]]
        inputs[model] = {
            "experiment": spec["experiment"], "fold_profile": spec["fold_profile"],
            "utterance_labels": utt_source, "phoneme_labels": phn_source,
            "utterance_order_sha256": spec["utterance_order_sha256"],
            "phoneme_ids": phone_ids.tolist(),
            "global_phoneme_presence_counts": global_phone.tolist(),
            "global_utt_mean_histogram_counts": global_score.tolist(),
            "global_utt_mean": float(scores.mean()),
        }
        for number, train, validation in fold_indices(groups, profile["validation_speakers"]):
            for split, indices in (("train", train), ("validation", validation)):
                phone = presence[indices].sum(axis=0)
                score = np.histogram(scores[indices], bins=edges)[0]
                rows.append({
                    "model": model, "experiment": spec["experiment"],
                    "fold": number, "subset": split,
                    "utterances": len(indices), "speakers": len(set(groups[indices])),
                    "phoneme_overlap_percent": overlap_percent(global_phone, phone),
                    "utt_mean_overlap_percent": overlap_percent(global_score, score),
                    "phoneme_jensen_shannon_distance": float(jensenshannon(global_phone, phone, base=2)),
                    "utt_mean_wasserstein_points": float(wasserstein_distance(scores, scores[indices])),
                    "utt_mean": float(scores[indices].mean()),
                    "utt_mean_absolute_shift": float(abs(scores[indices].mean() - scores.mean())),
                    "phoneme_presence_counts": phone.tolist(),
                    "utt_mean_histogram_counts": score.tolist(),
                })
    report = {
        "scope": "BOMA only; six models, five folds per model",
        "reference": "Each model's full canonical training partition (2,500 utterances)",
        "similarity_definition": "100 * sum(min(p_i, q_i)); distributions normalized to sum to 1",
        "interpretation": "Histogram overlap percentage; not a probability of equivalence or a guarantee of stratification",
        "phoneme_convention": "Per-utterance presence counts of the 39 scored canonical phonemes, then normalized across phonemes",
        "utt_mean_convention": "Arithmetic mean of the five original [0,10] utterance annotations; rounded to six decimals before binning",
        "utt_mean_bin_edges": edges.tolist(),
        "utt_mean_intervals": "Left-closed, right-open; final interval includes 10",
        "aggregation": "Arithmetic mean across model-fold comparisons; equal model and fold weights; reused folds are not independent replications",
        "fold_assignment_status": assignments["status"],
        "fold_assignment_sha256": sha256(ASSIGNMENTS),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "inputs": inputs,
        "aggregate": {split: summarize([r for r in rows if r["subset"] == split]) for split in ("train", "validation")},
        "per_model": {
            model: {split: summarize([r for r in rows if r["model"] == model and r["subset"] == split]) for split in ("train", "validation")}
            for model in assignments["models"]
        },
        "fold_results": rows,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", type=Path,
                        help="Root containing restored model folders; repeat to search several roots. Defaults to the repository.")
    parser.add_argument("--utt-bins", type=int, default=20, help="Equal-width UTT-Mean bins over [0,10] (default: 20).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Write summary.json and fold_similarity.csv here.")
    args = parser.parse_args()
    try:
        rows, report = calculate(args.data_root or [ROOT], args.utt_bins)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    fields = [key for key in rows[0] if not key.endswith("_counts")]
    with (args.output_dir / "fold_similarity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Similarity to full training: mean [minimum, maximum], percent")
    for split, values in report["aggregate"].items():
        for metric in ("phoneme_overlap_percent", "utt_mean_overlap_percent"):
            v = values[metric]
            print(f"{split:10} {metric:26} {v['mean']:.4f} [{v['minimum']:.4f}, {v['maximum']:.4f}] (n={values['comparisons']})")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
