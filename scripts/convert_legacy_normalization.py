"""Convert recovered MATLAB normalization files to the repository NPZ format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import FEATURE_LAYOUTS, NormalizationStats, get_feature_layout


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Legacy model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-layout",
        choices=sorted(FEATURE_LAYOUTS),
        default="final_checkpoint_9spatial",
    )
    return parser.parse_args()


def scalar(mapping: dict, key: str) -> float:
    return float(np.asarray(mapping[key]).squeeze())


def main() -> None:
    arguments = parse_arguments()
    layout = get_feature_layout(arguments.feature_layout)
    source = arguments.source
    input_values = loadmat(source / "input_normalization.mat", simplify_cells=True)
    temperature = loadmat(source / "output_normalization_t.mat", simplify_cells=True)
    salinity = loadmat(source / "output_normalization_s.mat", simplify_cells=True)
    pressure = loadmat(source / "pressure_normalization.mat", simplify_cells=True)

    feature_count = len(layout.feature_indices)
    input_mean = np.asarray(
        [scalar(input_values, f"mu_{index}") for index in range(1, feature_count + 1)]
    )
    input_std = np.asarray(
        [scalar(input_values, f"sig_{index}") for index in range(1, feature_count + 1)]
    )
    output_mean = np.column_stack(
        (
            [scalar(temperature, f"mu_{index}") for index in range(1, 14)],
            [scalar(salinity, f"mu_{index}") for index in range(1, 14)],
        )
    )
    output_std = np.column_stack(
        (
            [scalar(temperature, f"sig_{index}") for index in range(1, 14)],
            [scalar(salinity, f"sig_{index}") for index in range(1, 14)],
        )
    )
    stats = NormalizationStats(
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        pressure_mean=scalar(pressure, "mu_pres"),
        pressure_std=scalar(pressure, "sigma_pres"),
        layout_name=layout.name,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    stats.save(arguments.output)


if __name__ == "__main__":
    main()
