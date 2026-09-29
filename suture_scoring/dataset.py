"""PyTorch dataset over the suturing images and their annotations.

Augmentation randomness is governed by the global torch RNG, which the training
loop seeds via ``set_seed`` — the dataset itself takes no seed (a per-dataset
seed would not control torchvision's transforms anyway).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .data import SCORE_KEYS, Annotation
from .preprocess import preprocess_image, to_tensor


class SutureDataset(Dataset):
    """One item per annotated image: (3, size, size) float tensor in [0, 1] + target scores."""

    def __init__(
        self,
        image_dir: str | Path,
        annotations: List[Annotation],
        size: int = 224,
        augment: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.items = [(a.name, a.scores) for a in annotations]
        self.size = size
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, scores = self.items[index]
        image = preprocess_image(self.image_dir / name, size=self.size)
        tensor = to_tensor(image)
        if self.transform is not None:
            tensor = self.transform(tensor)
        targets = {key: torch.tensor(scores[key]) for key in SCORE_KEYS}
        return tensor, targets


def split_annotations(annotations: List[Annotation], frac: float = 0.12, seed: int = 0):
    """Deterministic shuffled split: (train, val) annotation lists; val holds ~frac."""
    rng = random.Random(seed)
    shuffled = list(annotations)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * frac))
    return shuffled[n_val:], shuffled[:n_val]