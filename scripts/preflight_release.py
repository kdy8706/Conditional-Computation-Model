"""Validate the documented v0.1.1 source and v0.1.0 asset layout before running CCM."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import load_mat_dataset


EXPECTED_FILE_SHA256 = {
    "model_epoch_996.pth": "79b6a661d77eca4fb8ca8c7d978a530e54234e8ea77a8b0941c09e1887766e02",
    "input_normalization.mat": "2b6bd902b7de525948afbc16975ced29ad0b9445c628a3cc99351a4b9b7addc1",
    "output_normalization_t.mat": "30d761e096ecc95c29a66b458ce51bb0330949dd470f4850bee72c39b75bffc6",
    "output_normalization_s.mat": "9c553d24c6027f69dec58184de72df8af944b5afeb7e5aeeef522cc7d12068c7",
    "pressure_normalization.mat": "19dc7edff54d5d5a54e941ff8fdf76d814ad1fc0d02dfd6c4bb964430f635327",
}


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
    for name in EXPECTED_FILE_SHA256:
        require(execution / name)
    train_path = archive / "trainset.mat"
    test_path = archive / "testset.mat"
    require(train_path)
    require(test_path)

    for name, expected_sha256 in EXPECTED_FILE_SHA256.items():
        actual_sha256 = sha256(execution / name)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Unexpected checksum for {name}: {actual_sha256}; expected {expected_sha256}"
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

