#!/usr/bin/env python
"""Scan decision thresholds for a Region-light checkpoint on a validation set."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from data.datasets import AVLip
    from lightweight.models import LipFDRegionLight

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = LipFDRegionLight("ViT-L/14", args.backbone)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    opt = argparse.Namespace(data_label="val", real_list_path=args.real_list_path, fake_list_path=args.fake_list_path)
    dataset = AVLip(opt)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    y_true, y_score = [], []
    with torch.inference_mode():
        for img, crops, label in tqdm(loader, desc="scoring"):
            img = img.to(device, non_blocking=True)
            crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
            score = model(crops, model.get_features(img))[0].sigmoid().flatten()
            y_score.extend(score.cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.linspace(0.01, 0.99, 197)
    rows = [compute_metrics(y_true, y_score, th) for th in thresholds]
    best_acc = max(rows, key=lambda x: x["acc"])
    best_f1 = max(rows, key=lambda x: x["f1"])
    at_05 = compute_metrics(y_true, y_score, 0.5)

    summary = {
        "ckpt": args.ckpt,
        "backbone": args.backbone,
        "score_summary": {
            "real_mean": float(y_score[y_true == 0].mean()),
            "fake_mean": float(y_score[y_true == 1].mean()),
            "real_p50": float(np.percentile(y_score[y_true == 0], 50)),
            "fake_p50": float(np.percentile(y_score[y_true == 1], 50)),
        },
        "threshold_0_5": at_05,
        "best_acc": best_acc,
        "best_f1": best_f1,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "thresholds": rows}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
