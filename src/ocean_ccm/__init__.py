"""Conditional computation for ocean subsurface profile reconstruction."""

from .data import (
    DEPTH_LEVELS_M,
    FEATURE_LAYOUTS,
    FINAL_CHECKPOINT_LAYOUT,
    PREVIOUS_DEPTH_LEVELS_M,
    TAKE5_LAYOUT,
    FeatureLayout,
    NormalizationStats,
    ProcessedDataset,
    RawDataset,
    get_feature_layout,
    load_mat_dataset,
    split_indices,
)

__all__ = [
    "DEPTH_LEVELS_M",
    "FEATURE_LAYOUTS",
    "FINAL_CHECKPOINT_LAYOUT",
    "PREVIOUS_DEPTH_LEVELS_M",
    "TAKE5_LAYOUT",
    "FeatureLayout",
    "NormalizationStats",
    "ProcessedDataset",
    "RawDataset",
    "get_feature_layout",
    "load_mat_dataset",
    "split_indices",
]

try:
    from .model import ConditionalModel, GatedConv2D, collect_outputs

    __all__ += ["ConditionalModel", "GatedConv2D", "collect_outputs"]
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
