"""Lightweight Region Awareness backbones for LipFD.

This module keeps the original Region Awareness forward interface unchanged:
    forward(crops, global_feature) -> pred_score, weights_max, weights_org

Only the ResNet depth is changed. The data semantics and crop layout are not
changed.
"""

from typing import Any

import torch.nn as nn

from models.region_awareness import BasicBlock, Bottleneck, _get_backbone


BACKBONE_CONFIGS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}


def get_region_backbone(
    arch: str = "resnet18",
    pretrained: bool = False,
    global_feature_dim: int = 768,
    **kwargs: Any,
):
    if arch not in BACKBONE_CONFIGS:
        raise ValueError(f"Unsupported backbone '{arch}'. Choose from {sorted(BACKBONE_CONFIGS)}.")

    block, layers = BACKBONE_CONFIGS[arch]
    model = _get_backbone(
        arch=arch,
        block=block,
        layers=layers,
        pretrained=pretrained,
        progress=True,
        **kwargs,
    )
    if global_feature_dim != 768:
        region_feature_dim = model.fc.in_features - 768
        in_features = region_feature_dim + global_feature_dim
        model.get_weight = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        model.fc = nn.Linear(in_features, 1)
    return model
