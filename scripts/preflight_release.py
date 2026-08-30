"""Validate the documented v0.1.0 Release layout before running CCM."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import load_mat_dataset


EXPECTED_CHECKPOINT_SHA256 = (
    "79b6a661d77eca4fb8ca8c7d978a530e54234e8ea77a8b0941c09e1887766e02"
)
NORMALIZATION_FILES = (
    "input_normalization.mat",
    "output_normalization_t.mat",
    "output_normalization_s.mat",
    "pressure_normalization.mat",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("artifacts/release"),
        help="Directory containing execution/ and archive/13/",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Release file: {path}")


def main() -> None:
    root = parse_arguments().release_root
    execution = root / "execution"
    archive = root / "archive" / "13"
    checkpoint = execution / "model_epoch_996.pth"
    require(checkpoint)
    for name in NORMALIZATION_FILES:
        require(execution / name)
    train_path = archive / "trainset.mat"
    test_path = archive / "testset.mat"
    require(train_path)
    require(test_path)

    actual_sha256 = sha256(checkpoint)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Unexpected checkpoint checksum: {actual_sha256}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    shape = tuple(state_dict["normal_gated_conv1.conv_feature.weight"].shape)
    if shape != (20, 10, 3, 3):
        raise ValueError(f"Expected take5 first-convolution shape (20, 10, 3, 3), got {shape}")

    for label, path in (("train", train_path), ("test", test_path)):
        data = load_mat_dataset(path)
        if data.inputs.shape[2] != 14:
            raise ValueError(f"Expected 14-channel {label} input, got {data.inputs.shape}")
        print(f"{label}: inputs={data.inputs.shape} targets={data.targets.shape}")
    print("Release preflight passed: verified take5 checkpoint and archive split layout.")


if __name__ == "__main__":
    main()

