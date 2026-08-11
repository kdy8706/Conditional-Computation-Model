"""Depth-wise metrics for temperature and salinity profiles."""

from __future__ import annotations

import numpy as np


def profile_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate metrics for arrays shaped ``(sample, depth, variable)``."""

    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 3:
        raise ValueError("observed and predicted must have matching (N, D, V) shapes")

    valid = np.isfinite(observed) & np.isfinite(predicted)
    masked_observed = np.where(valid, observed, np.nan)
    masked_predicted = np.where(valid, predicted, np.nan)
    error = masked_predicted - masked_observed

    rmse = np.sqrt(np.nanmean(error**2, axis=0))
    mae = np.nanmean(np.abs(error), axis=0)
    observed_mean = np.nanmean(masked_observed, axis=0)
    residual_sum = np.nansum(error**2, axis=0)
    total_sum = np.nansum((masked_observed - observed_mean[None, :, :]) ** 2, axis=0)
    r2 = 1.0 - np.divide(
        residual_sum,
        total_sum,
        out=np.full_like(residual_sum, np.nan),
        where=total_sum > 0,
    )
    observed_range = np.nanmax(masked_observed, axis=0) - np.nanmin(masked_observed, axis=0)
    nrmse = np.divide(
        rmse,
        observed_range,
        out=np.full_like(rmse, np.nan),
        where=observed_range > 0,
    )
    count = valid.sum(axis=0)
    return {"r2": r2, "rmse": rmse, "mae": mae, "nrmse": nrmse, "count": count}
