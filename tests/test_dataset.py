import torch

from suture_scoring.data import SCORE_KEYS, load_annotations
from suture_scoring.dataset import SutureDataset, split_annotations


def test_dataset_length_and_item_shapes(fixture_annotations, fixture_images_dir):
    annotations = load_annotations(fixture_annotations)
    ds = SutureDataset(fixture_images_dir, annotations, size=128)
    assert len(ds) == 3
    image, targets = ds[0]
    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert set(targets) == set(SCORE_KEYS)
    for key in SCORE_KEYS:
        assert 0 <= targets[key].item() <= 10


def test_dataset_augment_keeps_shapes(fixture_annotations, fixture_images_dir):
    annotations = load_annotations(fixture_annotations)
    ds = SutureDataset(fixture_images_dir, annotations, size=128, augment=True)
    image, targets = ds[0]
    assert image.shape == (3, 128, 128)
    assert set(targets) == set(SCORE_KEYS)


def test_split_annotations_deterministic_and_disjoint(fixture_annotations):
    annotations = load_annotations(fixture_annotations)
    train_a, val_a = split_annotations(annotations, frac=0.34, seed=7)
    train_b, val_b = split_annotations(annotations, frac=0.34, seed=7)
    assert [a.name for a in train_a] == [a.name for a in train_b]
    assert [a.name for a in val_a] == [a.name for a in val_b]
    train_names = {a.name for a in train_a}
    val_names = {a.name for a in val_a}
    assert train_names.isdisjoint(val_names)
    assert len(train_names) + len(val_names) == len(annotations)
    assert len(val_a) == 1  # round(3 * 0.34) = 1


def test_split_annotations_different_seeds_differ():
    annotations = _fake_annotations(20)
    _, val_a = split_annotations(annotations, frac=0.3, seed=1)
    _, val_b = split_annotations(annotations, frac=0.3, seed=2)
    assert [a.name for a in val_a] != [a.name for a in val_b]


def _fake_annotations(n):
    from suture_scoring.data import Annotation

    scores = {"Overall": 5, "ISD": 5, "Slack": 5, "Position": 5, "Angulation": 5, "Width": 5}
    return [Annotation(name=f"Image_{i:04d}_04_0_0_4.png", scores=scores) for i in range(n)]