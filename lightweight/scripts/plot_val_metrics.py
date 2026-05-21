#!/usr/bin/env python
"""Plot validation metric curves from a region-light history.json file."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        default=None,
        help="Path to history.json. If omitted, use --run_dir/history.json.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Training output directory, e.g. lightweight/results/checkpoints/region_resnet18_full_ra1.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    if args.history is None:
        if args.run_dir is None:
            raise ValueError("Provide either --history or --run_dir.")
        history_path = Path(args.run_dir) / "history.json"
    else:
        history_path = Path(args.history)

    output_path = Path(args.output) if args.output else history_path.with_name("val_metrics_curve.png")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    metric_history = [h for h in history if h.get("metrics")]
    if not metric_history:
        raise RuntimeError(f"No validation metrics found in {history_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in metric_history]
    plt.figure(figsize=(10, 6))
    for key, label in [
        ("acc", "Accuracy"),
        ("ap", "AP"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
    ]:
        values = [h["metrics"][key] for h in metric_history]
        plt.plot(epochs, values, marker="o", linewidth=2, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Region-light LipFD Validation Metrics")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved validation metrics curve: {output_path}")


if __name__ == "__main__":
    main()
