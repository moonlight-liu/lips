#!/usr/bin/env python
"""Scan DataLoader settings for end-to-end validation speed.

This script keeps the model/checkpoint fixed and tests combinations of:
    - num_workers
    - prefetch_factor

It uses the real validation dataset and writes one summary JSON/CSV at the end.
No per-batch files are written.
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight.scripts.benchmark_region_light_e2e import build_loader, load_model, run_benchmark


def make_run_args(base_args, num_workers, prefetch_factor):
    return argparse.Namespace(
        ckpt=base_args.ckpt,
        clip_name=base_args.clip_name,
        backbone=base_args.backbone,
        real_list_path=base_args.real_list_path,
        fake_list_path=base_args.fake_list_path,
        batch_size=base_args.batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=base_args.persistent_workers,
        gpu=base_args.gpu,
        threshold=base_args.threshold,
        max_batches=base_args.max_batches,
        output=None,
        save_scores=None,
    )


def flatten_row(result):
    metrics = result["metrics"]
    timing = result["timing"]
    return {
        "num_workers": result["num_workers"],
        "prefetch_factor": result["prefetch_factor"],
        "persistent_workers": result["persistent_workers"],
        "batch_size": result["batch_size"],
        "num_samples": timing["num_samples"],
        "num_batches": timing["num_batches"],
        "fps_including_metrics": timing["fps_including_metrics"],
        "total_ms_per_sample": timing["total_ms_per_sample"],
        "data_wait_ms_per_sample": timing["data_wait_ms_per_sample"],
        "transfer_forward_ms_per_sample": timing["transfer_forward_ms_per_sample"],
        "data_wait_seconds": timing["data_wait_seconds"],
        "transfer_forward_seconds": timing["transfer_forward_seconds"],
        "total_seconds": timing["total_seconds_including_metrics"],
        "acc": metrics["acc"],
        "ap": metrics["ap"],
        "fpr": metrics["fpr"],
        "fnr": metrics["fnr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--clip_name", default=None, choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers_list", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--prefetch_factors", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--output_json", default="./lightweight/results/checkpoints/dataloader_e2e_scan.json")
    parser.add_argument("--output_csv", default="./lightweight/results/checkpoints/dataloader_e2e_scan.csv")
    args = parser.parse_args()

    if args.max_batches < 0:
        args.max_batches = None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, clip_name, backbone = load_model(args, device)
    print(f"Device: {device}")
    print(f"CLIP: {clip_name}, backbone: {backbone}, global feature dim: {model.global_feature_dim}")

    results = {
        "ckpt": args.ckpt,
        "clip": clip_name,
        "backbone": backbone,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "runs": [],
    }

    for num_workers in args.num_workers_list:
        factors = args.prefetch_factors if num_workers > 0 else [None]
        for prefetch_factor in factors:
            run_args = make_run_args(args, num_workers, prefetch_factor if prefetch_factor is not None else 2)
            label = f"workers={num_workers}, prefetch={prefetch_factor if num_workers > 0 else 'none'}"
            print(f"\n== {label} ==")
            dataset, loader = build_loader(run_args, device)
            metrics, timing, batch_rows = run_benchmark(model, loader, dataset, device, run_args)
            row = {
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor if num_workers > 0 else None,
                "persistent_workers": args.persistent_workers if num_workers > 0 else False,
                "batch_size": args.batch_size,
                "metrics": metrics,
                "timing": timing,
                "batch_timing_summary": {
                    "first_batch": batch_rows[0] if batch_rows else None,
                    "last_batch": batch_rows[-1] if batch_rows else None,
                },
            }
            results["runs"].append(row)
            print(
                f"{label}: fps={timing['fps_including_metrics']:.2f}, "
                f"total={timing['total_ms_per_sample']:.2f} ms/sample, "
                f"data={timing['data_wait_ms_per_sample']:.2f} ms/sample, "
                f"forward={timing['transfer_forward_ms_per_sample']:.2f} ms/sample"
            )
            del loader
            del dataset
            gc.collect()

    flat_rows = [flatten_row(r) for r in results["runs"]]
    best = max(flat_rows, key=lambda r: r["fps_including_metrics"]) if flat_rows else None
    results["best_by_fps"] = best

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if flat_rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    print("\n== best by fps ==")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"saved json: {output_json}")
    print(f"saved csv: {output_csv}")


if __name__ == "__main__":
    main()
