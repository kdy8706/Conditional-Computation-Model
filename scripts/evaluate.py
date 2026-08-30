"""Evaluate a legacy or refactored CCM checkpoint on a MATLAB dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocean_ccm.data import FEATURE_LAYOUTS, NormalizationStats, get_feature_layout, load_mat_dataset
from ocean_ccm.metrics import profile_metrics
from ocean_ccm.model import ConditionalModel, collect_outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path)
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
    parser.add_argument(
        "--feature-layout",
        choices=sorted(FEATURE_LAYOUTS),
        help="Defaults to the layout stored in the normalization file.",
    )
    return parser.parse_args()


def json_ready(metrics: dict[str, np.ndarray]) -> dict[str, list]:
    ready = {}
    for name, values in metrics.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.floating):
            array = np.where(np.isfinite(array), array, None)
        ready[name] = array.tolist()
    return ready


def main() -> None:
    arguments = parse_arguments()
    raw = load_mat_dataset(arguments.data)
    stats = NormalizationStats.load(arguments.normalization)
    layout = get_feature_layout(arguments.feature_layout or stats.layout_name)
    processed = stats.transform(
        raw,
        layout=layout,
        vector_grid_index=tuple(arguments.vector_grid_index),
        missing_value_policy=arguments.missing_value_policy,
    )

    dataset = TensorDataset(
        torch.from_numpy(processed.spatial),
        torch.from_numpy(processed.vector),
        torch.from_numpy(processed.previous_pressure),
        torch.from_numpy(processed.eddy_signal),
    )
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalModel(
        input_shape_spatial=(len(layout.spatial_indices), 8, 8),
        mask_mode=arguments.mask_mode,
        pool_mode=arguments.pool_mode,
    ).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictions = []
    with torch.no_grad():
        for spatial, vector, pressure, signal in loader:
            spatial, vector, pressure, signal = [
                value.to(device) for value in (spatial, vector, pressure, signal)
            ]
            grouped = model(spatial, vector, pressure, signal)
            ordered = collect_outputs(grouped, spatial.shape[0], reference=spatial)
            predictions.append(ordered.cpu().numpy())

    normalized_prediction = np.concatenate(predictions, axis=0)
    prediction = stats.inverse_targets(normalized_prediction)
    metrics = profile_metrics(raw.targets, prediction)
    result = {
        "temperature": json_ready({name: value[:, 0] for name, value in metrics.items()}),
        "salinity": json_ready({name: value[:, 1] for name, value in metrics.items()}),
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
