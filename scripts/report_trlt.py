"""Rebuild action metrics from exported predictions, or plot seed summaries (host only)."""

import argparse
import gzip
import json

import numpy as np

from openpi.trlt.cache import write_json
from openpi.trlt.config import digest
from openpi.trlt.config import root_path
from openpi.trlt.metrics import ActionMetrics
from openpi.trlt.metrics import summarize_results


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rebuild = sub.add_parser("rebuild")
    rebuild.add_argument("--predictions", required=True)
    rebuild.add_argument("--normalization", required=True)
    rebuild.add_argument("--output", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("metrics", nargs="+")
    plot.add_argument("--output-dir", default="experiments/trlt/summary")
    args = parser.parse_args()
    if args.command == "rebuild":
        normalization = json.loads(root_path(args.normalization).read_text())
        if normalization.get("hash") != digest({key: value for key, value in normalization.items() if key != "hash"}):
            raise ValueError("Normalization content hash mismatch")
        stats = normalization["action"]
        metrics = ActionMetrics(stats["mean"], stats["std"])
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        current_key, rows = None, []

        def accumulate(query_rows):
            if not query_rows:
                return
            horizon = max(row["horizon_offset"] for row in query_rows) + 1
            prediction = np.zeros((1, horizon, 58), np.float64)
            target = np.zeros_like(prediction)
            valid = np.zeros((1, horizon), bool)
            for row in query_rows:
                offset = row["horizon_offset"]
                prediction[0, offset] = (np.asarray(row["predicted_action"]) - mean) / std
                target[0, offset] = (np.asarray(row["gt_action"]) - mean) / std
                valid[0, offset] = row["valid"]
            metrics.update(prediction, target, valid, [query_rows[0]["episode_id"]], [query_rows[0]["object"]])

        with gzip.open(root_path(args.predictions), "rt") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["episode_id"], row["query_frame_index"])
                if current_key is not None and key != current_key:
                    accumulate(rows)
                    rows = []
                current_key = key
                rows.append(row)
        accumulate(rows)
        write_json(root_path(args.output), metrics.result())
        return
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    output = root_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results([root_path(path) for path in args.metrics], output / "summary.json")
    labels, means, stds = [], [], []
    lines = ["model,seeds,mean,std"]
    for model, values in summary["models"].items():
        lines.append(f"{model},{len(values['seeds'])},{values['mean']},{values['std']}")
        if values["mean"] is not None:
            labels.append(model)
            means.append(values["mean"])
            stds.append(values["std"] or 0.0)
    (output / "summary.csv").write_text("\n".join(lines) + "\n")
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, means, yerr=stds, capsize=4)
    axis.set_ylabel("Normalized action MSE (episode macro)")
    axis.tick_params(axis="x", labelrotation=15)
    axis.set_title("Offline action probe: all available seeds, mean ± sample std")
    fig.tight_layout()
    fig.savefig(output / "action_mse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
