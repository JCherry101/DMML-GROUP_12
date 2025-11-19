import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models


def upgrade_multi_head_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename legacy Sequential head keys (foo_head.0.weight) to Linear keys."""
    upgraded = copy.deepcopy(state_dict)
    for head in ("maker_head", "body_head", "color_head", "model_head"):
        weight_key = f"{head}.0.weight"
        bias_key = f"{head}.0.bias"
        if weight_key in upgraded:
            upgraded[f"{head}.weight"] = upgraded.pop(weight_key)
        if bias_key in upgraded:
            upgraded[f"{head}.bias"] = upgraded.pop(bias_key)
    return upgraded


class DualHeadCarNet(nn.Module):
    """EfficientNet backbone with optional color/model heads."""

    def __init__(
        self,
        num_makers: int,
        num_bodytypes: int,
        num_colors: Optional[int] = None,
        num_models: Optional[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception as e:
            print(f"[WARN] Failed to load pretrained EfficientNet weights ({e}); using random init.")
            self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.maker_head = nn.Linear(in_features, num_makers)
        self.body_head = nn.Linear(in_features, num_bodytypes)
        self.has_color_head = num_colors is not None and num_colors > 0
        self.color_head = nn.Linear(in_features, num_colors) if self.has_color_head else None
        self.has_model_head = num_models is not None and num_models > 0
        self.model_head = nn.Linear(in_features, num_models) if self.has_model_head else None

    def forward(self, x):
        feats = self.backbone(x)
        maker_logits = self.maker_head(feats)
        body_logits = self.body_head(feats)
        color_logits = self.color_head(feats) if self.color_head is not None else None
        model_logits = self.model_head(feats) if self.model_head is not None else None
        return maker_logits, body_logits, color_logits, model_logits

    def predict(self, x):
        maker_logits, body_logits, color_logits, model_logits = self.forward(x)
        maker_probs = torch.softmax(maker_logits, dim=1)
        body_probs = torch.softmax(body_logits, dim=1)
        color_probs = torch.softmax(color_logits, dim=1) if color_logits is not None else None
        model_probs = torch.softmax(model_logits, dim=1) if model_logits is not None else None
        return maker_probs, body_probs, color_probs, model_probs


if __name__ == "__main__":
    model = DualHeadCarNet(num_makers=10, num_bodytypes=5, num_colors=8, num_models=6, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    print([out.shape if out is not None else None for out in outputs])
