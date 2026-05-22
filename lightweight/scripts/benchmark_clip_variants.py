#!/usr/bin/env python
"""Measure CLIP variant size and synthetic speed without training.

This script is a smoke test for the next lightweight stage. It measures:
1. CLIP encoder parameters and image encoding speed.
2. Full Region-light forward speed when the CLIP feature dimension is compatible.

If a smaller CLIP variant cannot connect to the existing Region Awareness branch,
the script records the shape error instead of failing the whole run.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_call(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def make_full_inputs(batch_size, device):
    img = torch.randn(batch_size, 3, 1120, 1120, device=device)
    crops = [[torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(5)] for _ in range(3)]
    return img, crops


@torch.inference_mode()
def measure_variant(clip_name, backbone, batch_sizes, device, warmup, iters, full_forward):
    from lightweight.models import LipFDRegionLight

    model = LipFDRegionLight(clip_name=clip_name, backbone=backbone)
    model.to(device).eval()

    total_params = count_params(model)
    encoder_params = count_params(model.encoder)
    backbone_params = count_params(model.backbone)
    conv1_params = count_params(model.conv1)

    result = {
        "clip": clip_name,
        "backbone": backbone,
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "encoder_params": encoder_params,
        "encoder_params_m": encoder_params / 1e6,
        "backbone_params": backbone_params,
        "backbone_params_m": backbone_params / 1e6,
        "conv1_params": conv1_params,
        "batch_results": [],
    }

    for batch_size in batch_sizes:
        img, crops = make_full_inputs(batch_size, device)
        encoded = model.get_features(img)
        clip_encode = bench_call(lambda: model.get_features(img), device, warmup, iters)
        row = {
            "batch_size": batch_size,
            "feature_shape": list(encoded.shape),
            "clip_encode": {
                **clip_encode,
                "latency_ms_per_sample": clip_encode["mean_ms"] / batch_size,
                "throughput_samples_per_s": batch_size * 1000.0 / clip_encode["mean_ms"],
            },
        }

        if full_forward:
            try:
                _ = model(crops, encoded)
                full = bench_call(lambda: model(crops, model.get_features(img)), device, warmup, iters)
                row["full_forward"] = {
                    **full,
                    "latency_ms_per_sample": full["mean_ms"] / batch_size,
                    "throughput_samples_per_s": batch_size * 1000.0 / full["mean_ms"],
                }
            except Exception as exc:
                row["full_forward_error"] = f"{type(exc).__name__}: {exc}"

        result["batch_results"].append(row)

    return result


def print_summary(result):
    print(
        f"{result['clip']} params total={result['total_params_m']:.2f}M "
        f"encoder={result['encoder_params_m']:.2f}M backbone={result['backbone_params_m']:.2f}M"
    )
    for row in result["batch_results"]:
        clip = row["clip_encode"]
        msg = (
            f"  bs={row['batch_size']:<3} feature_shape={row['feature_shape']} "
            f"clip_encode={clip['mean_ms']:.2f} ms "
            f"({clip['latency_ms_per_sample']:.2f} ms/sample, "
            f"{clip['throughput_samples_per_s']:.2f} samples/s)"
        )
        if "full_forward" in row:
            full = row["full_forward"]
            msg += (
                f" full={full['mean_ms']:.2f} ms "
                f"({full['latency_ms_per_sample']:.2f} ms/sample, "
                f"{full['throughput_samples_per_s']:.2f} samples/s)"
            )
        if "full_forward_error" in row:
            msg += f" full_forward_error={row['full_forward_error']}"
        print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", default=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--full_forward", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    results = {
        "device": str(device),
        "backbone": args.backbone,
        "warmup": args.warmup,
        "iters": args.iters,
        "variants": [],
    }

    for clip_name in args.clips:
        result = measure_variant(
            clip_name=clip_name,
            backbone=args.backbone,
            batch_sizes=args.batch_sizes,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            full_forward=args.full_forward,
        )
        results["variants"].append(result)
        print_summary(result)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
