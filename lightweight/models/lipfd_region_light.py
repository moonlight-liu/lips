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
        self.global_feature_dim = self._infer_global_feature_dim()
        self.backbone = get_region_backbone(backbone, global_feature_dim=self.global_feature_dim)

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

    def _infer_global_feature_dim(self):
        output_dim = getattr(getattr(self.encoder, "visual", None), "output_dim", None)
        if output_dim is not None:
            return int(output_dim)
        known_dims = {
            "ViT-L/14": 768,
            "ViT-B/16": 512,
            "ViT-B/32": 512,
        }
        if self.clip_name in known_dims:
            return known_dims[self.clip_name]
        raise ValueError(f"Cannot infer CLIP feature dimension for {self.clip_name}")


def load_global_weights(model: LipFDRegionLight, ckpt_path: str):
    """Load matching conv1 and CLIP encoder weights from an original LipFD checkpoint.

    ViT-B variants cannot reuse ViT-L/14 encoder tensors from the official
    checkpoint because their shapes differ. Those keys are skipped while conv1
    is still reused.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    current_state = model.state_dict()

    global_state = OrderedDict()
    skipped_shape = []
    for key, value in state_dict.items():
        if key.startswith("conv1.") or key.startswith("encoder."):
            if key in current_state and current_state[key].shape == value.shape:
                global_state[key] = value
            elif key in current_state:
                skipped_shape.append((key, tuple(value.shape), tuple(current_state[key].shape)))

    missing, unexpected = model.load_state_dict(global_state, strict=False)
    loaded = len(global_state)
    return {
        "loaded_keys": loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "skipped_shape_keys": skipped_shape,
    }
