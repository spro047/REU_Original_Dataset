import numpy as np
import torch

from suture_scoring.data import SCORE_KEYS, load_annotations
from suture_scoring.dataset import split_annotations
from suture_scoring.train import load_model, train_model


def test_train_one_epoch_smoke(fixture_annotations, fixture_images_dir, tmp_path):
    annotations = load_annotations(fixture_annotations)
    train_anns, val_anns = split_annotations(annotations, frac=0.34, seed=0)
    checkpoint = tmp_path / "model.pt"

    history, best_mae = train_model(
        fixture_images_dir,
        train_anns,
        val_anns,
        epochs=1,
        batch_size=2,
        size=64,
        backbone="tiny",
        embedding=16,
        seed=0,
        checkpoint=checkpoint,
        verbose=False,
    )

    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0.0
    assert 1 in history["val"]
    val_metrics = history["val"][1]
    assert set(val_metrics) == set(SCORE_KEYS)
    for key in SCORE_KEYS:
        assert "mae" in val_metrics[key]
        assert val_metrics[key]["mae"] >= 0.0
    assert best_mae == val_metrics["Overall"]["mae"]

    state = torch.load(checkpoint, weights_only=True)
    assert "state_dict" in state
    assert state["config"]["backbone"] == "tiny"


def test_load_model_rebuilds_and_scores(fixture_annotations, fixture_images_dir, tmp_path):
    annotations = load_annotations(fixture_annotations)
    train_anns, val_anns = split_annotations(annotations, frac=0.34, seed=0)
    checkpoint = tmp_path / "model.pt"
    train_model(
        fixture_images_dir, train_anns, val_anns,
        epochs=1, batch_size=2, size=64, backbone="tiny", embedding=16,
        seed=0, checkpoint=checkpoint, verbose=False,
    )

    model = load_model(checkpoint)
    assert model.heads["Overall"].in_features == 16

    from suture_scoring.preprocess import to_tensor

    logits = model(to_tensor(np.zeros((64, 64, 3), dtype=np.float32)).unsqueeze(0))
    assert set(logits) == set(SCORE_KEYS)