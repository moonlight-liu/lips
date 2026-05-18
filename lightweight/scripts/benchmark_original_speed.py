#!/usr/bin/env python
"""Benchmark the original LipFD model speed.

Two modes are supported:
1. synthetic: model-only timing with random tensors already on the target device.
2. val: end-to-end validation loader timing on real preprocessed samples.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start, end):
    return (end - start) * 1000.0


def summarize_times(times_ms, batch_size):
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    p90_ms = sorted(times_ms)[int(0.9 * (len(times_ms) - 1))]
    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "p90_ms": p90_ms,
        "throughput_samples_per_s": batch_size * 1000.0 / mean_ms,
        "single_sample_latency_ms": mean_ms / batch_size,
    }


def make_synthetic_inputs(batch_size, device):
    img = torch.randn(batch_size, 3, 1120, 1120, device=device)
    crops = [
        [torch.randn(batch_size, 3, 224, 224, device=device) for _ in range(5)]
        for _ in range(3)
    ]
    return img, crops


@torch.inference_mode()
def benchmark_synthetic(model, device, batch_size, warmup, iters):
    img, crops = make_synthetic_inputs(batch_size, device)

    for _ in range(warmup):
        features = model.get_features(img)
        _ = model(crops, features)
    sync_if_needed(device)

    full_times = []
    feature_times = []
    backbone_times = []

    for _ in range(iters):
        sync_if_needed(device)
        start = time.perf_counter()
        features = model.get_features(img)
        sync_if_needed(device)
        feature_end = time.perf_counter()

        _ = model(crops, features)
        sync_if_needed(device)
        end = time.perf_counter()

        feature_times.append(elapsed_ms(start, feature_end))
        backbone_times.append(elapsed_ms(feature_end, end))
        full_times.append(elapsed_ms(start, end))

    result = summarize_times(full_times, batch_size)
    result["mode"] = "synthetic"
    result["feature_mean_ms"] = statistics.mean(feature_times)
    result["backbone_mean_ms"] = statistics.mean(backbone_times)
    return result


@torch.inference_mode()
def benchmark_val_loader(model, device, args, batch_size):
    from data.datasets import AVLip

    opt = argparse.Namespace(
        data_label="val",
        real_list_path=args.real_list_path,
        fake_list_path=args.fake_list_path,
    )
    dataset = AVLip(opt)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    total_samples = 0
    total_batches = 0
    model_compute_ms = []
    end_to_end_start = time.perf_counter()

    for batch_idx, (img, crops, label) in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        batch_start = time.perf_counter()
        img = img.to(device, non_blocking=True)
        crops = [[t.to(device, non_blocking=True) for t in sublist] for sublist in crops]
        sync_if_needed(device)
        compute_start = time.perf_counter()

        features = model.get_features(img)
        _ = model(crops, features)
        sync_if_needed(device)
        compute_end = time.perf_counter()

        model_compute_ms.append(elapsed_ms(compute_start, compute_end))
        total_samples += int(label.numel())
        total_batches += 1

        if args.print_every and total_batches % args.print_every == 0:
            elapsed = time.perf_counter() - end_to_end_start
            print(
                f"val batch {total_batches}: "
                f"{total_samples / max(elapsed, 1e-9):.2f} samples/s"
            )

        _ = batch_start

    end_to_end_sec = time.perf_counter() - end_to_end_start
    if total_batches == 0:
        raise RuntimeError("No validation batches were processed.")

    compute_summary = summarize_times(model_compute_ms, batch_size)
    return {
        "mode": "val",
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "batches": total_batches,
        "samples": total_samples,
        "end_to_end_seconds": end_to_end_sec,
        "end_to_end_samples_per_s": total_samples / end_to_end_sec,
        "end_to_end_ms_per_sample": end_to_end_sec * 1000.0 / total_samples,
        "model_compute_mean_ms_per_batch": compute_summary["mean_ms"],
        "model_compute_ms_per_sample": compute_summary["single_sample_latency_ms"],
    }


def print_synthetic_result(result):
    print(
        f"synthetic bs={result['batch_size']:<3} "
        f"full={result['mean_ms']:.2f} ms/batch "
        f"latency={result['single_sample_latency_ms']:.2f} ms/sample "
        f"throughput={result['throughput_samples_per_s']:.2f} samples/s "
        f"feature={result['feature_mean_ms']:.2f} ms "
        f"backbone={result['backbone_mean_ms']:.2f} ms"
    )


def print_val_result(result):
    print(
        f"val bs={result['batch_size']:<3} workers={result['num_workers']:<2} "
        f"batches={result['batches']} samples={result['samples']} "
        f"e2e={result['end_to_end_samples_per_s']:.2f} samples/s "
        f"e2e_latency={result['end_to_end_ms_per_sample']:.2f} ms/sample "
        f"model_latency={result['model_compute_ms_per_sample']:.2f} ms/sample"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark original LipFD speed.")
    parser.add_argument("--arch", default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", default="./checkpoints/ckpt.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["synthetic", "val", "both"], default="both")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--real_list_path", default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=100,
        help="Limit validation batches for quick benchmarking. Use -1 for full val.",
    )
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument(
        "--output",
        default="./lightweight/results/original_speed.json",
        help="Path to save the JSON result.",
    )
    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches < 0:
        args.max_batches = None

    from models import build_model

    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    model = build_model(args.arch)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()

    results = {
        "arch": args.arch,
        "ckpt": args.ckpt,
        "device": str(device),
        "batch_results": [],
    }

    for batch_size in args.batch_sizes:
        if args.mode in ("synthetic", "both"):
            result = benchmark_synthetic(model, device, batch_size, args.warmup, args.iters)
            results["batch_results"].append(result)
            print_synthetic_result(result)

        if args.mode in ("val", "both"):
            result = benchmark_val_loader(model, device, args, batch_size)
            results["batch_results"].append(result)
            print_val_result(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
