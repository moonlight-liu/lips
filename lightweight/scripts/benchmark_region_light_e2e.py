#!/usr/bin/env python
"""End-to-end validation speed benchmark for Region-light LipFD.

This benchmark uses the real validation dataset and includes:
    - DataLoader iteration and image loading
    - CPU -> GPU transfer
    - model forward
    - score collection
    - metric computation

It writes one JSON file at the end if --output is provided. No per-batch files
are written.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def load_model(args, device):
    from lightweight.models import LipFDRegionLight

    state = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    clip_name = args.clip_name or ckpt_args.get("clip_name", "ViT-L/14")
    backbone = args.backbone or ckpt_args.get("backbone", "resnet18")

    model = LipFDRegionLight(clip_name=clip_name, backbone=backbone)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    return model, clip_name, backbone


def build_loader(args, device):
    from data.datasets import AVLip

    opt = argparse.Namespace(
        data_label="val",
        real_list_path=args.real_list_path,
        fake_list_path=args.fake_list_path,
    )
    dataset = AVLip(opt)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


@torch.inference_mode()
def run_benchmark(model, loader, dataset, device, args):
    y_true, y_score, sample_paths = [], [], []
    batch_rows = []

    total_start = time.perf_counter()
    iter_start = time.perf_counter()
    loader_iter = iter(loader)
    data_wait_s = 0.0
    transfer_forward_s = 0.0
    score_collect_s = 0.0
    processed = 0
    batch_idx = 0

    while True:
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        data_start = time.perf_counter()
        try:
            img, crops, label = next(loader_iter)
        except StopIteration:
            break
        data_end = time.perf_counter()
        data_time = data_end - data_start
        data_wait_s += data_time

        forward_start = time.perf_counter()
        img = img.to(device, non_blocking=True)
        crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
        sync(device)
        score = model(crops, model.get_features(img))[0].sigmoid().flatten()
        sync(device)
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        transfer_forward_s += forward_time

        collect_start = time.perf_counter()
        score_list = score.cpu().tolist()
        label_list = label.flatten().tolist()
        y_score.extend(score_list)
        y_true.extend(label_list)
        start = batch_idx * args.batch_size
        end = start + int(label.numel())
        sample_paths.extend(dataset.total_list[start:end])
        collect_end = time.perf_counter()
        collect_time = collect_end - collect_start
        score_collect_s += collect_time

        batch_size = int(label.numel())
        processed += batch_size
        batch_rows.append(
            {
                "batch": batch_idx + 1,
                "batch_size": batch_size,
                "data_wait_ms": data_time * 1000.0,
                "transfer_forward_ms": forward_time * 1000.0,
                "score_collect_ms": collect_time * 1000.0,
            }
        )
        batch_idx += 1

    loop_end = time.perf_counter()
    metric_start = time.perf_counter()
    y_true_np = np.array(y_true)
    y_score_np = np.array(y_score)
    y_pred = (y_score_np >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "acc": float(accuracy_score(y_true_np, y_pred)),
        "ap": float(average_precision_score(y_true_np, y_score_np)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    metric_end = time.perf_counter()

    total_s = metric_end - total_start
    loop_s = loop_end - total_start
    metric_s = metric_end - metric_start
    overhead_s = max(0.0, loop_s - data_wait_s - transfer_forward_s - score_collect_s)

    timing = {
        "num_samples": int(processed),
        "num_batches": int(batch_idx),
        "total_seconds_including_metrics": total_s,
        "loop_seconds_excluding_metrics": loop_s,
        "metric_seconds": metric_s,
        "data_wait_seconds": data_wait_s,
        "transfer_forward_seconds": transfer_forward_s,
        "score_collect_seconds": score_collect_s,
        "other_loop_overhead_seconds": overhead_s,
        "seconds_per_sample": total_s / processed if processed else None,
        "fps_including_metrics": processed / total_s if total_s else None,
        "fps_excluding_metrics": processed / loop_s if loop_s else None,
        "data_wait_ms_per_sample": data_wait_s * 1000.0 / processed if processed else None,
        "transfer_forward_ms_per_sample": transfer_forward_s * 1000.0 / processed if processed else None,
        "total_ms_per_sample": total_s * 1000.0 / processed if processed else None,
    }

    if args.save_scores:
        scores_path = Path(args.save_scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "score"])
            writer.writerows(zip(sample_paths, y_true_np.tolist(), y_score_np.tolist()))

    return metrics, timing, batch_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--clip_name", default=None, choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_scores", default=None)
    args = parser.parse_args()

    if args.max_batches < 0:
        args.max_batches = None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, clip_name, backbone = load_model(args, device)
    dataset, loader = build_loader(args, device)

    print(f"Device: {device}")
    print(f"CLIP: {clip_name}, backbone: {backbone}, global feature dim: {model.global_feature_dim}")
    print(
        f"Dataset samples: {len(dataset)}, batch_size: {args.batch_size}, "
        f"workers: {args.num_workers}, prefetch_factor: {args.prefetch_factor}, "
        f"persistent_workers: {args.persistent_workers}"
    )

    metrics, timing, batch_rows = run_benchmark(model, loader, dataset, device, args)
    result = {
        "ckpt": args.ckpt,
        "clip": clip_name,
        "backbone": backbone,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
        "device": str(device),
        "params": {
            "total": count_params(model),
            "total_m": count_params(model) / 1e6,
            "encoder_m": count_params(model.encoder) / 1e6,
            "backbone_m": count_params(model.backbone) / 1e6,
        },
        "metrics": metrics,
        "timing": timing,
        "batch_timing_summary": {
            "first_batch": batch_rows[0] if batch_rows else None,
            "last_batch": batch_rows[-1] if batch_rows else None,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved benchmark: {output_path}")


if __name__ == "__main__":
    main()
