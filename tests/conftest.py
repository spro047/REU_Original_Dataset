from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_images_dir() -> Path:
    return FIXTURES / "images"


@pytest.fixture
def fixture_annotations() -> Path:
    return FIXTURES / "annotations.xlsx"