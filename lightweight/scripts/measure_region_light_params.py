#!/usr/bin/env python
"""Measure parameter counts for Region-light LipFD variants."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_name", default="ViT-L/14", choices=["ViT-L/14", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from lightweight.models import LipFDRegionLight

    model = LipFDRegionLight(clip_name=args.clip_name, backbone=args.backbone)
    total = count_params(model)
    encoder = count_params(model.encoder)
    backbone = count_params(model.backbone)
    conv1 = count_params(model.conv1)
    result = {
        "clip": args.clip_name,
        "backbone": args.backbone,
        "global_feature_dim": model.global_feature_dim,
        "total_params": total,
        "total_params_m": total / 1e6,
        "encoder_params": encoder,
        "encoder_params_m": encoder / 1e6,
        "backbone_params": backbone,
        "backbone_params_m": backbone / 1e6,
        "conv1_params": conv1,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
