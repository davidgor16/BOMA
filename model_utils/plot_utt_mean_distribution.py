"""Plot the shared training UTT-Mean histogram from the similarity report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "results/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    counts = [item["global_utt_mean_histogram_counts"] for item in report["inputs"].values()]
    if len(counts) != 6 or any(item != counts[0] for item in counts):
        parser.error("A shared figure requires identical training histograms for all six models.")
    edges = np.asarray(report["utt_mean_bin_edges"])
    if len(edges) != len(counts[0]) + 1 or sum(counts[0]) != 2500:
        parser.error("Unexpected training histogram dimensions or sample count.")
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.5, 2.2), layout="constrained")
    ax.bar(edges[:-1], counts[0], width=np.diff(edges), align="edge",
           color="#df5959", edgecolor="#222222", linewidth=0.65)
    ax.set(xlim=(0, 10), xlabel="UTT-Mean score", ylabel="Number of utterances")
    ax.set_xticks(np.arange(11))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#cccccc", linestyle="--", linewidth=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=450)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
