#!/usr/bin/env python
"""Scan fake-score decision thresholds from a saved CSV.

Expected CSV columns:
    path,label,score

This script does not load the model or images. It reads scores once, scans
thresholds in memory, then optionally writes one JSON result file.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def load_scores(path):
    labels = []
    scores = []
    sample_paths = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label", "score"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            sample_paths.append(row["path"])
            labels.append(int(float(row["label"])))
            scores.append(float(row["score"]))

    if not labels:
        raise ValueError(f"No scores found in {path}")
    return sample_paths, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def metrics_at_threshold(labels, scores, threshold):
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = int(labels.size)
    real_total = int((labels == 0).sum())
    fake_total = int((labels == 1).sum())

    return {
        "threshold": float(threshold),
        "acc": float((tp + tn) / total),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "fpr": float(fp / real_total) if real_total else 0.0,
        "fnr": float(fn / fake_total) if fake_total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="CSV saved by validate_region_light.py --save_scores")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--min_threshold", type=float, default=0.01)
    parser.add_argument("--max_threshold", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    args = parser.parse_args()

    sample_paths, labels, scores = load_scores(args.scores)
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step / 2.0, args.step)
    rows = [metrics_at_threshold(labels, scores, th) for th in thresholds]

    best_acc = max(rows, key=lambda x: (x["acc"], x["f1"], -x["fpr"]))
    best_f1 = max(rows, key=lambda x: (x["f1"], x["acc"], -x["fpr"]))
    at_05 = metrics_at_threshold(labels, scores, 0.5)
    fpr_candidates = [row for row in rows if row["fpr"] <= args.target_fpr]
    best_under_target_fpr = max(fpr_candidates, key=lambda x: (x["acc"], x["f1"])) if fpr_candidates else None

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    summary = {
        "scores": args.scores,
        "num_samples": int(labels.size),
        "num_real": int(real_scores.size),
        "num_fake": int(fake_scores.size),
        "ap": float(average_precision_score(labels, scores)),
        "score_summary": {
            "real_mean": float(real_scores.mean()) if real_scores.size else None,
            "fake_mean": float(fake_scores.mean()) if fake_scores.size else None,
            "real_p50": float(np.percentile(real_scores, 50)) if real_scores.size else None,
            "fake_p50": float(np.percentile(fake_scores, 50)) if fake_scores.size else None,
            "real_p95": float(np.percentile(real_scores, 95)) if real_scores.size else None,
            "fake_p05": float(np.percentile(fake_scores, 5)) if fake_scores.size else None,
        },
        "threshold_0_5": at_05,
        "best_acc": best_acc,
        "best_f1": best_f1,
        f"best_acc_with_fpr_le_{args.target_fpr:g}": best_under_target_fpr,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "thresholds": rows}, f, indent=2, ensure_ascii=False)
        print(f"saved threshold scan: {output_path}")


if __name__ == "__main__":
    main()
