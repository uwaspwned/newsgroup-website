from pathlib import Path

import pytest

from app.config import Config
from app.model_verification import verify_model


def model_artifacts_exist() -> bool:
    return all([
        Config.MODEL_PATH.exists(),
        Path(f"{Config.MODEL_PATH}.sig").exists(),
        Config.PUBLIC_KEY_PATH.exists(),
    ])


@pytest.mark.skipif(not model_artifacts_exist(), reason="Model artifacts are not generated")
def test_model_signature_is_valid():
    assert verify_model() is True
