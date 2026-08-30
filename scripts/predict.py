"""Generate denormalized CCM profiles and save them as a MATLAB file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import DEPTH_LEVELS_M, NormalizationStats, load_mat_dataset
from ocean_ccm.model import ConditionalModel, collect_outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mask-mode", choices=["legacy", "zero_invalid"], default="legacy")
    parser.add_argument("--pool-mode", choices=["max", "avg"], default="max")
    parser.add_argument(
        "--vector-grid-index",
        type=int,
        nargs=2,
        metavar=("ROW", "COLUMN"),
        default=(3, 3),
    )
    parser.add_argument(
        "--missing-value-policy",
        choices=["preserve", "sentinel"],
        default="preserve",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    raw = load_mat_dataset(arguments.data)
    stats = NormalizationStats.load(arguments.normalization)
    processed = stats.transform(
        raw,
        vector_grid_index=tuple(arguments.vector_grid_index),
        missing_value_policy=arguments.missing_value_policy,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(processed.spatial),
            torch.from_numpy(processed.vector),
            torch.from_numpy(processed.previous_pressure),
            torch.from_numpy(processed.eddy_signal),
        ),
        batch_size=arguments.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalModel(
        input_shape_spatial=(processed.spatial.shape[1], 8, 8),
        mask_mode=arguments.mask_mode,
        pool_mode=arguments.pool_mode,
    ).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()

    normalized = []
    with torch.no_grad():
        for batch in loader:
            spatial, vector, pressure, signal = [value.to(device) for value in batch]
            grouped = model(spatial, vector, pressure, signal)
            normalized.append(
                collect_outputs(grouped, spatial.shape[0], reference=spatial).cpu().numpy()
            )

    prediction = stats.inverse_targets(np.concatenate(normalized, axis=0))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        arguments.output,
        {
            "prediction": prediction,
            "temperature": prediction[:, :, 0],
            "salinity": prediction[:, :, 1],
            "depth_m": DEPTH_LEVELS_M,
        },
    )


if __name__ == "__main__":
    main()
