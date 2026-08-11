"""Optional real-artifact regression test.

Set all four CCM_EPOCH996_* variables to run this locally. They are excluded
from Git because they point to restricted research artifacts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REQUIRED = (
    "CCM_EPOCH996_DATA",
    "CCM_EPOCH996_CHECKPOINT",
    "CCM_EPOCH996_NORMALIZATION",
    "CCM_EPOCH996_REFERENCE",
)


@pytest.mark.skipif(
    any(not os.environ.get(name) for name in REQUIRED),
    reason="Set CCM_EPOCH996_* artifact paths to run the local regression test",
)
def test_epoch996_prediction_matches_saved_result():
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "verify_epoch996_regression.py"),
        "--data",
        os.environ["CCM_EPOCH996_DATA"],
        "--checkpoint",
        os.environ["CCM_EPOCH996_CHECKPOINT"],
        "--normalization",
        os.environ["CCM_EPOCH996_NORMALIZATION"],
        "--reference",
        os.environ["CCM_EPOCH996_REFERENCE"],
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
