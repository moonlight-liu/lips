#!/usr/bin/env python
"""Synthetic speed benchmark for Region-light LipFD variants."""

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
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(batch_size, device):
    img = torch.randn(batch_size, 3, 1120, 1120, device=device)
    crops = [[torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(5)] for _ in range(3)]
    return img, crops


@torch.inference_mode()
def bench(model, device, batch_size, warmup, iters):
    img, crops = make_inputs(batch_size, device)
    for _ in range(warmup):
        _ = model(crops, model.get_features(img))
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        _ = model(crops, model.get_features(img))
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.mean(times)
    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times),
        "throughput_samples_per_s": batch_size * 1000.0 / mean_ms,
        "latency_ms_per_sample": mean_ms / batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--ckpt", default=None, help="Optional trained region-light checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from lightweight.models import LipFDRegionLight

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = LipFDRegionLight(clip_name="ViT-L/14", backbone=args.backbone)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    results = {"backbone": args.backbone, "device": str(device), "batch_results": []}
    for bs in args.batch_sizes:
        result = bench(model, device, bs, args.warmup, args.iters)
        results["batch_results"].append(result)
        print(
            f"{args.backbone} bs={bs:<3} mean={result['mean_ms']:.2f} ms "
            f"latency={result['latency_ms_per_sample']:.2f} ms/sample "
            f"throughput={result['throughput_samples_per_s']:.2f} samples/s"
        )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
