"""Python equivalents of the deterministic post-matching MATLAB steps.

The product-specific download and file-matching stages remain outside this
module.  These functions cover patch extraction, missing-data screening,
outlier filtering, and OW-based routing-channel construction from the recovered
``ch3_patch.mlx`` and ``ch4_nan_processing.mlx`` workflow.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


TAKE5_SPATIAL_ORDER = (
    "ssh",
    "surface_wind_u",
    "surface_wind_v",
    "tidal_elevation",
    "tidal_current_u",
    "tidal_current_v",
    "net_heat_flux",
    "longitude",
    "latitude",
    "bathymetry",
)


def extract_centered_patches(
    field: np.ndarray,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    *,
    patch_size: int = 8,
) -> np.ndarray:
    """Extract the asymmetric even-sized patches used by the MATLAB code.

    ``field`` must be ``(rows, columns, samples)`` and indices are zero-based.
    For an 8 x 8 patch, the center index has three cells before it and four
    after it, matching MATLAB's ``index-3:index+4`` slice.
    """

    values = np.asarray(field)
    rows = np.asarray(row_indices, dtype=np.int64)
    columns = np.asarray(column_indices, dtype=np.int64)
    if values.ndim != 3:
        raise ValueError(f"Expected field (rows, columns, samples), got {values.shape}")
    if rows.shape != columns.shape or rows.ndim != 1:
        raise ValueError("row_indices and column_indices must be equal-length vectors")
    if values.shape[2] != rows.size:
        raise ValueError("Field sample count and index count differ")
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer")

    before = patch_size // 2 - 1
    after = patch_size // 2
    patches = np.empty((patch_size, patch_size, rows.size), dtype=values.dtype)
    for sample, (row, column) in enumerate(zip(rows, columns, strict=True)):
        row_slice = slice(row - before, row + after + 1)
        column_slice = slice(column - before, column + after + 1)
        patch = values[row_slice, column_slice, sample]
        if patch.shape != (patch_size, patch_size):
            raise ValueError(
                f"Sample {sample} at ({row}, {column}) falls outside the field boundary"
            )
        patches[:, :, sample] = patch
    return patches


def stack_take5_inputs(
    spatial: Mapping[str, np.ndarray],
    sst: np.ndarray,
    sss: np.ndarray,
    day_of_year: np.ndarray,
    eddy_signal: np.ndarray,
) -> np.ndarray:
    """Build the 14-channel ``take5`` tensor in documented channel order."""

    missing = [name for name in TAKE5_SPATIAL_ORDER if name not in spatial]
    if missing:
        raise KeyError(f"Missing spatial fields: {', '.join(missing)}")
    patch_arrays = [np.asarray(spatial[name]) for name in TAKE5_SPATIAL_ORDER]
    reference_shape = patch_arrays[0].shape
    if len(reference_shape) != 3 or reference_shape[:2] != (8, 8):
        raise ValueError(f"Expected spatial patches (8, 8, N), got {reference_shape}")
    if any(array.shape != reference_shape for array in patch_arrays):
        raise ValueError("All spatial fields must have the same shape")

    sample_count = reference_shape[2]
    vectors = [np.asarray(value).reshape(-1) for value in (sst, sss, day_of_year)]
    if any(value.size != sample_count for value in vectors):
        raise ValueError("Point-variable sample counts do not match the spatial fields")
    signal = np.asarray(eddy_signal)
    if signal.shape == (sample_count,):
        signal = np.broadcast_to(signal[None, None, :], reference_shape)
    if signal.shape != reference_shape:
        raise ValueError(f"Expected eddy signal (8, 8, N) or (N,), got {signal.shape}")

    point_fields = [np.broadcast_to(value[None, None, :], reference_shape) for value in vectors]
    return np.stack(patch_arrays + point_fields + [signal], axis=2)


def valid_patch_mask(inputs: np.ndarray, *, minimum_valid_cells: int = 32) -> np.ndarray:
    """Return samples having at least 32 fully observed grid cells.

    This reproduces the MATLAB rule: a cell is valid only when every channel at
    that cell is finite, and a sample survives when at least half of its 64
    cells are valid.
    """

    values = np.asarray(inputs)
    if values.ndim != 4 or values.shape[:2] != (8, 8):
        raise ValueError(f"Expected inputs (8, 8, C, N), got {values.shape}")
    valid_cells = np.isfinite(values).all(axis=2)
    return valid_cells.sum(axis=(0, 1)) >= minimum_valid_cells


def within_zscore_bounds(
    values: np.ndarray,
    *,
    z_limit: float = 2.56,
    axis: tuple[int, ...] | int = 0,
) -> np.ndarray:
    """Return sample mask excluding values outside mean +/- z_limit * std."""

    array = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(array, axis=axis, keepdims=True)
    std = np.nanstd(array, axis=axis, keepdims=True, ddof=1)
    inside = (array >= mean - z_limit * std) & (array <= mean + z_limit * std)
    reduced_axes = tuple(range(array.ndim - 1))
    return np.logical_or(inside, np.isnan(array)).all(axis=reduced_axes)


def ow_eddy_mask(ow_center: np.ndarray, ow_std: np.ndarray, *, factor: float = -0.2) -> np.ndarray:
    """Apply the recovered eddy criterion ``OW < -0.2 * OW_std``."""

    ow = np.asarray(ow_center, dtype=np.float64).reshape(-1)
    scale = np.asarray(ow_std, dtype=np.float64).reshape(-1)
    if ow.shape != scale.shape:
        raise ValueError("ow_center and ow_std must have the same sample count")
    return ow < factor * scale


def remove_nhf_for_final_checkpoint(take5_inputs: np.ndarray) -> np.ndarray:
    """Convert a 14-channel take5 tensor to the 13-channel epoch-996 layout."""

    values = np.asarray(take5_inputs)
    if values.ndim != 4 or values.shape[:3] != (8, 8, 14):
        raise ValueError(f"Expected take5 inputs (8, 8, 14, N), got {values.shape}")
    return np.delete(values, 6, axis=2)
