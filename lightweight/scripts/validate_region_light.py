#!/usr/bin/env python
"""Validate a Region-light LipFD checkpoint."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_scores", default=None, help="Optional CSV path for path,label,score. Written once after validation.")
    args = parser.parse_args()

    from data.datasets import AVLip
    from lightweight.models import LipFDRegionLight

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = LipFDRegionLight(clip_name="ViT-L/14", backbone=args.backbone)
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

    y_true, y_score, sample_paths = [], [], []
    with torch.inference_mode():
        for batch_idx, (img, crops, label) in enumerate(tqdm(loader, desc="validating")):
            img = img.to(device, non_blocking=True)
            crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
            score = model(crops, model.get_features(img))[0].sigmoid().flatten()
            y_score.extend(score.cpu().tolist())
            y_true.extend(label.flatten().tolist())
            start = batch_idx * args.batch_size
            end = start + int(label.numel())
            sample_paths.extend(dataset.total_list[start:end])

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_score)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")

    if args.save_scores:
        scores_path = Path(args.save_scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "score"])
            writer.writerows(zip(sample_paths, y_true.tolist(), y_score.tolist()))
        print(f"saved scores: {scores_path}")


if __name__ == "__main__":
    main()
