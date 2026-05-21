"""LipFD variant that only replaces the Region Awareness ResNet branch."""

from collections import OrderedDict

import torch
import torch.nn as nn

from models.clip import clip
from .region_awareness_light import get_region_backbone


class LipFDRegionLight(nn.Module):
    def __init__(self, clip_name: str = "ViT-L/14", backbone: str = "resnet18"):
        super().__init__()
        self.clip_name = clip_name
        self.backbone_name = backbone
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)
        self.encoder, self.preprocess = clip.load(clip_name, device="cpu")
        self.backbone = get_region_backbone(backbone)

    def forward(self, crops, feature):
        return self.backbone(crops, feature)

    def get_features(self, x):
        x = self.conv1(x)
        return self.encoder.encode_image(x)

    def freeze_global_encoder(self):
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_global_weights(model: LipFDRegionLight, ckpt_path: str):
    """Load only conv1 and CLIP encoder weights from an original LipFD checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    global_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("conv1.") or key.startswith("encoder."):
            global_state[key] = value

    missing, unexpected = model.load_state_dict(global_state, strict=False)
    loaded = len(global_state)
    return {
        "loaded_keys": loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
