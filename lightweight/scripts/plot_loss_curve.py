#!/usr/bin/env python
"""Plot training loss curves from a region-light training directory."""

import argparse
import csv
from pathlib import Path


def moving_average(values, window):
    if window <= 1:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start : i + 1]) / (i - start + 1))
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to loss_history.csv. If omitted, use --run_dir/loss_history.csv.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Training output directory, e.g. lightweight/results/checkpoints/region_resnet18_stage1.",
    )
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window.")
    args = parser.parse_args()

    if args.csv is None:
        if args.run_dir is None:
            raise ValueError("Provide either --csv or --run_dir.")
        csv_path = Path(args.run_dir) / "loss_history.csv"
    else:
        csv_path = Path(args.csv)

    if args.output is None:
        output_path = csv_path.with_name("loss_curve.png")
    else:
        output_path = Path(args.output)

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(r["step"]) for r in rows]
    loss = moving_average([float(r["loss"]) for r in rows], args.smooth)
    cls_loss = moving_average([float(r["cls_loss"]) for r in rows], args.smooth)
    ra_loss = moving_average([float(r["ra_loss"]) for r in rows], args.smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss, label="total loss", linewidth=1.8)
    plt.plot(steps, cls_loss, label="classification loss", linewidth=1.2)
    plt.plot(steps, ra_loss, label="region awareness loss", linewidth=1.2)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Region-light LipFD Training Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved loss curve: {output_path}")


if __name__ == "__main__":
    main()
