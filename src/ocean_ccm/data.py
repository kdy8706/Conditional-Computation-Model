"""MATLAB dataset loading and legacy-compatible preprocessing.

Two input layouts were recovered from the author-provided artifacts.  The
selected ``model_epoch_996.pth`` checkpoint uses 13 raw channels (nine spatial,
three point variables, and one routing channel).  The later ``take5/cnn.py``
uses 14 raw channels because it adds net heat flux to the spatial encoder.
Both are explicit here so that a checkpoint is never evaluated with the wrong
channel order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.io import loadmat


DEPTH_LEVELS_M = np.asarray(
    [10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500],
    dtype=np.float64,
)
PREVIOUS_DEPTH_LEVELS_M = np.asarray(
    [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400],
    dtype=np.float64,
)

FINAL_SPATIAL_CHANNELS = (
    "ssh",
    "surface_wind_1",
    "surface_wind_2",
    "tidal_elevation",
    "tidal_current_1",
    "tidal_current_2",
    "longitude",
    "latitude",
    "bathymetry",
)
TAKE5_SPATIAL_CHANNELS = FINAL_SPATIAL_CHANNELS[:6] + (
    "net_heat_flux",
) + FINAL_SPATIAL_CHANNELS[6:]
VECTOR_CHANNELS = ("sst", "sss", "day_of_year")
MissingValuePolicy = Literal["sentinel", "preserve"]


@dataclass(frozen=True)
class FeatureLayout:
    """Channel indices for one recovered input-tensor version."""

    name: str
    spatial_indices: tuple[int, ...]
    vector_indices: tuple[int, ...]
    eddy_index: int

    @property
    def feature_indices(self) -> tuple[int, ...]:
        return self.spatial_indices + self.vector_indices

    @property
    def spatial_names(self) -> tuple[str, ...]:
        if self.name == "final_checkpoint_9spatial":
            return FINAL_SPATIAL_CHANNELS
        return TAKE5_SPATIAL_CHANNELS


FINAL_CHECKPOINT_LAYOUT = FeatureLayout(
    name="final_checkpoint_9spatial",
    spatial_indices=tuple(range(9)),
    vector_indices=(9, 10, 11),
    eddy_index=12,
)
TAKE5_LAYOUT = FeatureLayout(
    name="take5_10spatial",
    spatial_indices=tuple(range(10)),
    vector_indices=(10, 11, 12),
    eddy_index=13,
)
FEATURE_LAYOUTS = {
    FINAL_CHECKPOINT_LAYOUT.name: FINAL_CHECKPOINT_LAYOUT,
    TAKE5_LAYOUT.name: TAKE5_LAYOUT,
}


def get_feature_layout(name: str) -> FeatureLayout:
    try:
        return FEATURE_LAYOUTS[name]
    except KeyError as error:
        choices = ", ".join(sorted(FEATURE_LAYOUTS))
        raise ValueError(f"Unknown feature layout {name!r}; choose one of: {choices}") from error


@dataclass(frozen=True)
class RawDataset:
    """Raw arrays in their MATLAB layout."""

    inputs: np.ndarray
    targets: np.ndarray
    pressure: np.ndarray

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])

    def validate(self) -> None:
        if self.inputs.ndim != 4 or self.inputs.shape[:2] != (8, 8):
            raise ValueError(f"Expected inputs (8, 8, C, N), got {self.inputs.shape}")
        if self.inputs.shape[2] not in (13, 14):
            raise ValueError(
                "Expected a recovered 13- or 14-channel input layout, "
                f"got {self.inputs.shape[2]} channels"
            )
        if self.targets.shape != (self.size, 13, 2):
            raise ValueError(f"Expected targets (N, 13, 2), got {self.targets.shape}")
        if self.pressure.shape != (14, self.size):
            raise ValueError(f"Expected pressure (14, N), got {self.pressure.shape}")
        if self.inputs.shape[-1] != self.size:
            raise ValueError("Input and target sample counts differ")


@dataclass(frozen=True)
class ProcessedDataset:
    """Model-ready NumPy arrays."""

    spatial: np.ndarray
    vector: np.ndarray
    previous_pressure: np.ndarray
    eddy_signal: np.ndarray
    targets: np.ndarray

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    pressure_mean: float
    pressure_std: float
    layout_name: str = FINAL_CHECKPOINT_LAYOUT.name

    @classmethod
    def fit(
        cls,
        data: RawDataset,
        indices: Sequence[int],
        *,
        layout: FeatureLayout = FINAL_CHECKPOINT_LAYOUT,
    ) -> "NormalizationStats":
        data.validate()
        if max(layout.feature_indices + (layout.eddy_index,)) >= data.inputs.shape[2]:
            raise ValueError(
                f"Layout {layout.name!r} is incompatible with {data.inputs.shape[2]} channels"
            )
        index = np.asarray(indices, dtype=np.int64)
        x = np.take(data.inputs[:, :, :, index], layout.feature_indices, axis=2)
        y = data.targets[index]
        pressure = data.pressure[:, index]

        input_mean = np.nanmean(x, axis=(0, 1, 3))
        input_std = np.nanstd(x, axis=(0, 1, 3))
        output_mean = np.nanmean(y, axis=0)
        output_std = np.nanstd(y, axis=0)
        pressure_mean = float(np.nanmean(pressure))
        pressure_std = float(np.nanstd(pressure))

        input_std = np.where(input_std == 0, 1.0, input_std)
        output_std = np.where(output_std == 0, 1.0, output_std)
        if pressure_std == 0:
            pressure_std = 1.0

        return cls(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
            layout_name=layout.name,
        )

    def transform(
        self,
        data: RawDataset,
        indices: Sequence[int] | None = None,
        *,
        layout: FeatureLayout | None = None,
        vector_grid_index: tuple[int, int] = (3, 3),
        missing_value: float = -999.0,
        missing_value_policy: MissingValuePolicy = "sentinel",
    ) -> ProcessedDataset:
        data.validate()
        layout = get_feature_layout(self.layout_name) if layout is None else layout
        if layout.name != self.layout_name:
            raise ValueError(
                f"Normalization statistics use {self.layout_name!r}, not {layout.name!r}"
            )
        index = np.arange(data.size) if indices is None else np.asarray(indices, dtype=np.int64)

        x = np.array(data.inputs[:, :, :, index], dtype=np.float64, copy=True)
        y = np.array(data.targets[index], dtype=np.float64, copy=True)
        pressure = np.array(data.pressure[:, index], dtype=np.float64, copy=True)

        feature_indices = layout.feature_indices
        if self.input_mean.shape != (len(feature_indices),):
            raise ValueError(
                f"Expected {len(feature_indices)} input statistics for {layout.name!r}, "
                f"got {self.input_mean.shape}"
            )
        selected = np.take(x, feature_indices, axis=2)
        normalized = (
            selected - self.input_mean[None, None, :, None]
        ) / self.input_std[None, None, :, None]
        y = (y - self.output_mean[None, :, :]) / self.output_std[None, :, :]
        pressure = (pressure - self.pressure_mean) / self.pressure_std

        spatial_count = len(layout.spatial_indices)
        eddy_signal = np.transpose(x[:, :, layout.eddy_index, :], (2, 0, 1))
        spatial = np.transpose(normalized[:, :, :spatial_count, :], (3, 2, 0, 1))
        row, column = vector_grid_index
        vector = np.transpose(normalized[row, column, spatial_count:, :], (1, 0))
        previous_pressure = pressure[:13, :].T

        arrays = [spatial, vector, previous_pressure, eddy_signal, y]
        if missing_value_policy == "sentinel":
            arrays = [np.nan_to_num(a, nan=missing_value) for a in arrays]
        elif missing_value_policy != "preserve":
            raise ValueError(
                f"Unknown missing-value policy: {missing_value_policy!r}"
            )
        arrays = [array.astype(np.float32) for array in arrays]
        return ProcessedDataset(*arrays)

    def inverse_targets(self, normalized_targets: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized_targets, dtype=np.float64)
        return values * self.output_std[None, :, :] + self.output_mean[None, :, :]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            input_mean=self.input_mean,
            input_std=self.input_std,
            output_mean=self.output_mean,
            output_std=self.output_std,
            pressure_mean=np.asarray(self.pressure_mean),
            pressure_std=np.asarray(self.pressure_std),
            layout_name=np.asarray(self.layout_name),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        with np.load(path) as values:
            if "layout_name" in values:
                layout_name = str(values["layout_name"])
            else:
                layout_name = (
                    TAKE5_LAYOUT.name
                    if len(values["input_mean"]) == len(TAKE5_LAYOUT.feature_indices)
                    else FINAL_CHECKPOINT_LAYOUT.name
                )
            return cls(
                input_mean=values["input_mean"],
                input_std=values["input_std"],
                output_mean=values["output_mean"],
                output_std=values["output_std"],
                pressure_mean=float(values["pressure_mean"]),
                pressure_std=float(values["pressure_std"]),
                layout_name=layout_name,
            )


def load_mat_dataset(paths: str | Path | Iterable[str | Path]) -> RawDataset:
    """Load standard or archived MATLAB profile datasets.

    Standard files expose ``Dinput2``, ``Doutput2``, and ``Dpressure2``.
    The released archive stores an already-created split with either
    ``xtrain``/``ytrain``/``pressure_train`` or
    ``xtest``/``ytest``/``pressure_test``.  Supporting both layouts keeps the
    archive executable without altering its historical arrays.
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]

    loaded: list[RawDataset] = []
    for path in paths:
        values = loadmat(path)
        if {"Dinput2", "Doutput2", "Dpressure2"}.issubset(values):
            inputs, targets, pressure = (
                values["Dinput2"],
                values["Doutput2"],
                values["Dpressure2"],
            )
        elif {"xtrain", "ytrain", "pressure_train"}.issubset(values):
            inputs, targets, pressure = (
                values["xtrain"],
                values["ytrain"],
                values["pressure_train"],
            )
        elif {"xtest", "ytest", "pressure_test"}.issubset(values):
            inputs, targets, pressure = (
                values["xtest"],
                values["ytest"],
                values["pressure_test"],
            )
        else:
            raise ValueError(
                f"{path} must contain Dinput2/Doutput2/Dpressure2, "
                "xtrain/ytrain/pressure_train, or xtest/ytest/pressure_test"
            )
        dataset = RawDataset(
            inputs=np.asarray(inputs),
            targets=np.asarray(targets),
            pressure=np.asarray(pressure),
        )
        dataset.validate()
        loaded.append(dataset)

    if not loaded:
        raise ValueError("At least one dataset path is required")
    if len(loaded) == 1:
        return loaded[0]

    combined = RawDataset(
        inputs=np.concatenate([item.inputs for item in loaded], axis=3),
        targets=np.concatenate([item.targets for item in loaded], axis=0),
        pressure=np.concatenate([item.pressure for item in loaded], axis=1),
    )
    combined.validate()
    return combined


def split_indices(
    sample_count: int,
    *,
    train_fraction: float = 0.8,
    seed: int = 20250217,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible shuffled train/validation split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(sample_count)
    train_count = int(np.floor(sample_count * train_fraction))
    return shuffled[:train_count], shuffled[train_count:]

