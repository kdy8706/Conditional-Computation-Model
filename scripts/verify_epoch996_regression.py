"""Verify that the cleaned legacy profile reproduces epoch-996 predictions.

The input data, checkpoint, normalization files, and historical MATLAB result
file stay outside Git.  This command compares generated profiles to the saved
``mod_val`` array and fails if their finite-value mask or values drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import NormalizationStats, load_mat_dataset
from ocean_ccm.model import ConditionalModel, collect_outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mean-absolute-tolerance", type=float, default=2e-4)
    parser.add_argument("--max-absolute-tolerance", type=float, default=5e-3)
    return parser.parse_args()


def predict(
    data,
    checkpoint: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data.spatial),
            torch.from_numpy(data.vector),
            torch.from_numpy(data.previous_pressure),
            torch.from_numpy(data.eddy_signal),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model = ConditionalModel(
        input_shape_spatial=(9, 8, 8),
        mask_mode="legacy",
        pool_mode="max",
    ).eval()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict), strict=True)

    outputs = []
    with torch.no_grad():
        for spatial, vector, pressure, eddy in loader:
            grouped = model(spatial, vector, pressure, eddy)
            outputs.append(collect_outputs(grouped, spatial.shape[0], reference=spatial).numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    arguments = parse_arguments()
    raw = load_mat_dataset(arguments.data)
    stats = NormalizationStats.load(arguments.normalization)
    if stats.layout_name != "final_checkpoint_9spatial":
        raise ValueError("Epoch-996 regression requires final_checkpoint_9spatial statistics")
    processed = stats.transform(
        raw,
        vector_grid_index=(0, 0),
        missing_value_policy="preserve",
    )
    prediction = stats.inverse_targets(
        predict(processed, arguments.checkpoint, batch_size=arguments.batch_size)
    )
    reference = np.asarray(loadmat(arguments.reference, simplify_cells=True)["mod_val"])
    if prediction.shape != reference.shape:
        raise AssertionError(f"Shape mismatch: generated {prediction.shape}, saved {reference.shape}")

    prediction_finite = np.isfinite(prediction)
    reference_finite = np.isfinite(reference)
    if not np.array_equal(prediction_finite, reference_finite):
        mismatches = int(np.count_nonzero(prediction_finite != reference_finite))
        raise AssertionError(f"Finite-value mask differs at {mismatches} positions")
    difference = np.abs(prediction[prediction_finite] - reference[prediction_finite])
    mean_absolute = float(np.mean(difference))
    max_absolute = float(np.max(difference))
    if mean_absolute > arguments.mean_absolute_tolerance:
        raise AssertionError(
            f"Mean absolute difference {mean_absolute} exceeds "
            f"{arguments.mean_absolute_tolerance}"
        )
    if max_absolute > arguments.max_absolute_tolerance:
        raise AssertionError(
            f"Maximum absolute difference {max_absolute} exceeds "
            f"{arguments.max_absolute_tolerance}"
        )
    print(
        f"epoch-996 regression passed: finite={difference.size}, "
        f"mean_abs={mean_absolute:.8g}, max_abs={max_absolute:.8g}"
    )


if __name__ == "__main__":
    main()
