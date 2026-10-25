"""
The pretrained backbones compared in Step 3, loaded from timm.

A FrameClassifier returns, for each face crop, a fake logit and the feature
vector the logit is computed from. The temporal models (Step 4) take sequences
of these feature vectors as input.
"""

import timm
import torch
from torch import nn

# name: timm model (input size, feature size)
BACKBONES = {
    "xception": "legacy_xception.tf_in1k",               # 299 px, 2048; the FF++ baseline
    "efficientnet_b4": "tf_efficientnet_b4.ap_in1k",      # 380 px, 1792; the weights SBI used
    "vit_b16": "vit_base_patch16_224.augreg_in21k",       # 224 px, 768; ImageNet-21k only
    "clip_vit_b16": "vit_base_patch16_clip_224.openai",   # 224 px, 768; CLIP's image encoder
}
CNNS = {"xception", "efficientnet_b4"}


class FrameClassifier(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        # num_classes=0: the backbone returns pooled features, and our own head scores them
        self.backbone = timm.create_model(BACKBONES[backbone], pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 1)
        # input size, mean and std the weights were trained with
        self.data_config = timm.data.resolve_model_data_config(self.backbone)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch,) fake logits and (batch, features) feature vectors."""
        features = self.backbone(x)
        return self.head(self.dropout(features)).squeeze(1), features


def set_trainable(model: FrameClassifier, mode: str) -> None:
    """full:   every weight is trained.
    linear: only the head; the backbone is frozen (a linear probe).
    norm:   the head and the backbone's normalisation layers, a light fine-tune
            that keeps most of a large pretrained model (e.g. CLIP) unchanged."""
    for p in model.backbone.parameters():
        p.requires_grad = mode == "full"
    if mode == "norm":
        for m in model.backbone.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                for p in m.parameters():
                    p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
