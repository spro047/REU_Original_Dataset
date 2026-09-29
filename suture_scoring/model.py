"""Scoring models.

Ticket 01 ships a deterministic placeholder scorer so the pipeline works end to
end and is reproducible. It is replaced by the real ordinal-regression model in
ticket 03; the scorer interface (image array in, score dict out) is the seam
that must stay stable.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import SCORE_KEYS
from .ordinal import NUM_CATEGORIES, ordinal_confidence, ordinal_score
from .preprocess import to_tensor


class ToyScorer:
    """Deterministic placeholder: scores derived from mean image brightness.

    Scores are mapped onto the 0-10 scale from the mean brightness of the
    normalized image; confidence is a fixed 0.9 placeholder. Deterministic for
    a given input, so CLI output is reproducible. Note the white padding
    (preprocess) biases brightness upward for non-square images — acceptable
    for a placeholder, replaced by the real model in ticket 03.
    """

    def score(self, image: np.ndarray) -> dict[str, float]:
        base = round(float(image.mean()) * 10.0, 1)
        out = {key: round(max(0.0, min(10.0, base + 0.1 * i)), 1) for i, key in enumerate(SCORE_KEYS)}
        out.update({f"conf_{key}": 0.9 for key in SCORE_KEYS})
        return out


class _TinyBackbone(nn.Module):
    """Small CNN for CPU development runs; the paper run uses resnet18."""

    def __init__(self, embedding: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embedding),
        )

    def forward(self, x):
        return self.features(x)


class _ResNet18Backbone(nn.Module):
    def __init__(self, embedding: int = 128, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet18

        self.net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.net.fc = nn.Identity()
        self.proj = nn.Linear(512, embedding)

    def forward(self, x):
        return self.proj(self.net(x))


def build_backbone(name: str, embedding: int, pretrained: bool = False) -> nn.Module:
    if name == "tiny":
        return _TinyBackbone(embedding=embedding)
    if name == "resnet18":
        return _ResNet18Backbone(embedding=embedding, pretrained=pretrained)
    raise ValueError(f"unknown backbone: {name!r}")


class OrdinalSutureModel(nn.Module):
    """Multi-head ordinal model: one image in, one (K-1)-logit head per score output."""

    def __init__(self, backbone: str = "tiny", embedding: int = 128, pretrained: bool = False):
        super().__init__()
        self.backbone = build_backbone(backbone, embedding, pretrained)
        self.heads = nn.ModuleDict(
            {key: nn.Linear(embedding, NUM_CATEGORIES - 1) for key in SCORE_KEYS}
        )

    def forward(self, x):
        features = self.backbone(x)
        return {key: head(features) for key, head in self.heads.items()}


class OrdinalSutureScorer:
    """Score a preprocessed image with a trained model; same interface as ToyScorer."""

    def __init__(self, model: OrdinalSutureModel):
        self.model = model.eval()

    def score(self, image: np.ndarray) -> dict[str, float]:
        tensor = to_tensor(image).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            logits = self.model(tensor)
        out: dict[str, float] = {}
        for key in SCORE_KEYS:
            head = logits[key][0]
            out[key] = float(ordinal_score(head))
            out[f"conf_{key}"] = float(ordinal_confidence(head))
        return out