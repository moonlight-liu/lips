#!/usr/bin/env python
"""Measure the original LipFD model size and parameter counts."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sizeof_file_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Measure parameter counts for the original LipFD model."
    )
    parser.add_argument("--arch", default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", default="./checkpoints/ckpt.pth")
    parser.add_argument(
        "--output",
        default="./lightweight/results/original_params.json",
        help="Path to save the JSON result.",
    )
    parser.add_argument(
        "--load_ckpt",
        action="store_true",
        help="Load checkpoint before measuring. Parameter counts are the same either way.",
    )
    args = parser.parse_args()

    from models import build_model

    model = build_model(args.arch)
    if args.load_ckpt:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict["model"])

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    encoder_params = count_params(model.encoder)
    backbone_params = count_params(model.backbone)
    conv1_params = count_params(model.conv1)

    result = {
        "arch": args.arch,
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "trainable_params": trainable_params,
        "trainable_params_m": trainable_params / 1e6,
        "encoder_params": encoder_params,
        "encoder_params_m": encoder_params / 1e6,
        "encoder_ratio": encoder_params / total_params,
        "backbone_params": backbone_params,
        "backbone_params_m": backbone_params / 1e6,
        "backbone_ratio": backbone_params / total_params,
        "conv1_params": conv1_params,
        "conv1_params_k": conv1_params / 1e3,
        "conv1_ratio": conv1_params / total_params,
        "checkpoint_path": args.ckpt,
        "checkpoint_size_mb": sizeof_file_mb(args.ckpt),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Original LipFD parameter report")
    print("=" * 40)
    print(f"Architecture: {result['arch']}")
    print(f"Total params:      {total_params:,} ({result['total_params_m']:.2f}M)")
    print(f"Trainable params:  {trainable_params:,} ({result['trainable_params_m']:.2f}M)")
    print(
        f"CLIP encoder:      {encoder_params:,} "
        f"({result['encoder_params_m']:.2f}M, {result['encoder_ratio'] * 100:.1f}%)"
    )
    print(
        f"Region backbone:   {backbone_params:,} "
        f"({result['backbone_params_m']:.2f}M, {result['backbone_ratio'] * 100:.1f}%)"
    )
    print(
        f"Conv1 downsample:  {conv1_params:,} "
        f"({result['conv1_params_k']:.2f}K, {result['conv1_ratio'] * 100:.4f}%)"
    )
    if result["checkpoint_size_mb"] is not None:
        print(f"Checkpoint size:   {result['checkpoint_size_mb']:.2f} MB")
    print(f"Saved JSON:        {output_path}")


if __name__ == "__main__":
    main()
