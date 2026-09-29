"""Ordinal regression helpers (CORAL: K ordered categories via K-1 binary tasks).

P(y > k) = sigmoid(logits[k]) for k = 0..K-2; class probabilities are the
telescoping differences, clamped to be non-negative and re-normalized so the
distribution always sums to 1. Confidence is 1 - normalized Shannon entropy of
the distribution: 1 for a peaked distribution, 0 for a uniform one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_CATEGORIES = 11  # scores 0..10


def ordinal_target(labels: torch.Tensor, num_categories: int = NUM_CATEGORIES) -> torch.Tensor:
    """Encode labels (0..K-1) as K-1 cumulative binary targets [y>0, y>1, ..., y>K-2]."""
    k = torch.arange(num_categories - 1, device=labels.device)
    return (labels.unsqueeze(-1) > k).float()


def ordinal_probs(logits: torch.Tensor, num_categories: int = NUM_CATEGORIES) -> torch.Tensor:
    """Convert (..., K-1) ordinal logits into a (..., K) distribution summing to 1."""
    cumulative = torch.sigmoid(logits)
    extended = torch.cat([torch.ones_like(cumulative[..., :1]), cumulative], dim=-1)
    probs = extended[..., :-1] - extended[..., 1:]
    probs = torch.cat([probs, extended[..., -1:]], dim=-1)
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


def ordinal_score(logits: torch.Tensor, num_categories: int = NUM_CATEGORIES) -> torch.Tensor:
    """Predicted category (argmax of the ordinal distribution)."""
    return ordinal_probs(logits, num_categories).argmax(dim=-1)


def ordinal_confidence(logits: torch.Tensor, num_categories: int = NUM_CATEGORIES) -> torch.Tensor:
    """1 - normalized Shannon entropy of the ordinal distribution (in [0, 1])."""
    probs = ordinal_probs(logits, num_categories)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    h_max = torch.log(torch.tensor(float(num_categories), dtype=logits.dtype))
    return 1.0 - entropy / h_max


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_categories: int = NUM_CATEGORIES) -> torch.Tensor:
    """Mean binary cross-entropy over the K-1 ordinal tasks."""
    targets = ordinal_target(labels, num_categories)
    return F.binary_cross_entropy_with_logits(logits, targets)