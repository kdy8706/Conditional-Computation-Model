"""Export depth-wise metrics from the recovered MATLAB result files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEPTHS = (10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccm", type=Path, required=True, help="CCM model_result.mat")
    parser.add_argument("--non-ccm", type=Path, required=True, help="Non-CCM model_result.mat")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path, model: str):
    result = loadmat(path, simplify_cells=True)
    artifact = str(result["model_name"])
    for dataset, suffix in (("held_out", "test"), ("independent", "val")):
        for variable, prefix in (("temperature", "t"), ("salinity", "s")):
            block = result[f"{prefix}_{suffix}"]
            for metric in ("R2", "RMSE", "MAE", "NRMSE"):
                values = np.asarray(block[metric], dtype=np.float64).reshape(-1)
                for depth, value in zip(DEPTHS, values, strict=True):
                    yield model, artifact, dataset, variable, metric, depth, value


def main() -> None:
    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "artifact", "dataset", "variable", "metric", "depth_m", "value"))
        writer.writerows(rows(arguments.ccm, "CCM"))
        writer.writerows(rows(arguments.non_ccm, "non-CCM"))


if __name__ == "__main__":
    main()
