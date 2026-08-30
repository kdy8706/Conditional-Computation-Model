import numpy as np
from scipy.io import savemat

from ocean_ccm.data import (
    FINAL_CHECKPOINT_LAYOUT,
    TAKE5_LAYOUT,
    NormalizationStats,
    RawDataset,
    split_indices,
    load_mat_dataset,
)


def synthetic_raw(sample_count: int = 12, *, channels: int = 14) -> RawDataset:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(8, 8, channels, sample_count))
    layout = TAKE5_LAYOUT if channels == 14 else FINAL_CHECKPOINT_LAYOUT
    for offset, channel in enumerate(layout.vector_indices):
        inputs[:, :, channel, :] = offset + np.arange(sample_count)[None, None, :]
    inputs[:, :, layout.eddy_index, :] = 0
    inputs[:, :, layout.eddy_index, sample_count // 2 :] = 1
    targets = rng.normal(size=(sample_count, 13, 2))
    pressure_levels = np.asarray(
        [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500]
    )
    pressure = np.repeat(pressure_levels[:, None], sample_count, axis=1)
    return RawDataset(inputs=inputs, targets=targets, pressure=pressure)


def test_split_is_reproducible_and_complete():
    first = split_indices(100, seed=11)
    second = split_indices(100, seed=11)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 80
    assert sorted(np.concatenate(first).tolist()) == list(range(100))


def test_take5_preprocessing_shapes_and_routing_channel():
    raw = synthetic_raw()
    training, _ = split_indices(raw.size, seed=3)
    stats = NormalizationStats.fit(raw, training, layout=TAKE5_LAYOUT)
    processed = stats.transform(raw)
    assert processed.spatial.shape == (12, 10, 8, 8)
    assert processed.vector.shape == (12, 3)
    assert processed.previous_pressure.shape == (12, 13)
    assert processed.eddy_signal.shape == (12, 8, 8)
    assert processed.targets.shape == (12, 13, 2)
    assert np.all(processed.eddy_signal[:6] == 0)
    assert np.all(processed.eddy_signal[6:] == 1)


def test_final_checkpoint_layout_has_nine_spatial_channels():
    raw = synthetic_raw(channels=13)
    stats = NormalizationStats.fit(raw, np.arange(raw.size), layout=FINAL_CHECKPOINT_LAYOUT)
    processed = stats.transform(raw)
    assert processed.spatial.shape == (12, 9, 8, 8)
    assert processed.vector.shape == (12, 3)
    assert np.all(processed.eddy_signal[:6] == 0)
    assert np.all(processed.eddy_signal[6:] == 1)


def test_preserve_policy_leaves_nan_inputs_untouched():
    raw = synthetic_raw(channels=13)
    raw.inputs[0, 0, 0, 0] = np.nan
    stats = NormalizationStats.fit(raw, np.arange(raw.size), layout=FINAL_CHECKPOINT_LAYOUT)
    processed = stats.transform(raw, missing_value_policy="preserve")
    assert np.isnan(processed.spatial[0, 0, 0, 0])


def test_target_round_trip():
    raw = synthetic_raw()
    stats = NormalizationStats.fit(raw, np.arange(raw.size), layout=TAKE5_LAYOUT)
    processed = stats.transform(raw)
    restored = stats.inverse_targets(processed.targets)
    assert np.allclose(restored, raw.targets, atol=1e-5)


def test_archive_train_and_test_split_layouts_are_loaded(tmp_path):
    raw = synthetic_raw(sample_count=4)
    train_path = tmp_path / "trainset.mat"
    test_path = tmp_path / "testset.mat"
    savemat(
        train_path,
        {"xtrain": raw.inputs, "ytrain": raw.targets, "pressure_train": raw.pressure},
    )
    savemat(
        test_path,
        {"xtest": raw.inputs, "ytest": raw.targets, "pressure_test": raw.pressure},
    )
    assert load_mat_dataset(train_path).inputs.shape == raw.inputs.shape
    assert load_mat_dataset(test_path).targets.shape == raw.targets.shape

