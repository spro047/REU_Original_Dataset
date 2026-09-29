"""Training loop for the ordinal suture model: CORAL loss, per-output metrics,
fixed seeds, and best-checkpoint saving. The official Val split stays untouched
here — model selection uses a held-out slice of the Train cohort.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics
from .data import SCORE_KEYS, Annotation
from .dataset import SutureDataset
from .model import OrdinalSutureModel
from .ordinal import coral_loss, ordinal_confidence, ordinal_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> Dict[str, Dict[str, float]]:
    """Per-output dev metrics on a loader: {key: {mae, exact, tol1, spearman, ece}}."""
    model.eval()
    preds: Dict[str, List[int]] = {key: [] for key in SCORE_KEYS}
    confs: Dict[str, List[float]] = {key: [] for key in SCORE_KEYS}
    targets: Dict[str, List[int]] = {key: [] for key in SCORE_KEYS}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            for key in SCORE_KEYS:
                head = logits[key]
                preds[key].extend(ordinal_score(head).tolist())
                confs[key].extend(ordinal_confidence(head).tolist())
                targets[key].extend(labels[key].tolist())

    out: Dict[str, Dict[str, float]] = {}
    for key in SCORE_KEYS:
        y, p = targets[key], preds[key]
        out[key] = {
            "mae": metrics.mae(y, p),
            "exact": metrics.exact_match_accuracy(y, p),
            "tol1": metrics.within_tolerance_accuracy(y, p, 1),
            "spearman": metrics.spearman(y, p),
            "ece": metrics.ece(confs[key], [1 if a == b else 0 for a, b in zip(y, p)]),
        }
    return out


def load_model(checkpoint: str | Path) -> OrdinalSutureModel:
    """Rebuild a trained model from a checkpoint, validating its config.

    Raises if the checkpoint's config or state dict does not match the model it
    reconstructs (e.g. a different backbone or embedding size).
    """
    state = torch.load(checkpoint, weights_only=True)
    config = state["config"]
    model = OrdinalSutureModel(
        backbone=config["backbone"],
        embedding=config["embedding"],
        pretrained=config["pretrained"],
    )
    model.load_state_dict(state["state_dict"])
    return model


def train_model(
    image_dir: str | Path,
    train_anns: List[Annotation],
    val_anns: List[Annotation],
    *,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    size: int = 224,
    backbone: str = "tiny",
    embedding: int = 128,
    pretrained: bool = False,
    seed: int = 0,
    checkpoint: str | Path | None = None,
    verbose: bool = True,
):
    """Train on `train_anns`, select on `val_anns`; save the best checkpoint.

    Returns (history, best_mae) where history["train_loss"] is the per-epoch
    loss and history["val"][epoch] is the per-output dev metrics for that epoch.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SutureDataset(image_dir, train_anns, size=size, augment=True)
    val_ds = SutureDataset(image_dir, val_anns, size=size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = OrdinalSutureModel(backbone=backbone, embedding=embedding, pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict = {"train_loss": [], "val": {}}
    best_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            logits = model(images)
            loss = sum(coral_loss(logits[key], labels[key].to(device)) for key in SCORE_KEYS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(total_loss / max(n_batches, 1))

        val_metrics = evaluate(model, val_loader, device)
        history["val"][epoch] = val_metrics
        epoch_mae = val_metrics["Overall"]["mae"]
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"epoch {epoch}/{epochs} loss={history['train_loss'][-1]:.4f} "
                f"val Overall MAE={epoch_mae:.3f} exact={val_metrics['Overall']['exact']:.3f}"
            )

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "config": {
                    "backbone": backbone,
                    "embedding": embedding,
                    "pretrained": pretrained,
                    "size": size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                },
            },
            checkpoint,
        )
        if verbose:
            print(f"checkpoint -> {checkpoint} (best Overall MAE {best_mae:.3f})")
    return history, best_mae