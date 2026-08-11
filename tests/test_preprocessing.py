import numpy as np

from ocean_ccm.preprocessing import (
    TAKE5_SPATIAL_ORDER,
    extract_centered_patches,
    ow_eddy_mask,
    remove_nhf_for_final_checkpoint,
    stack_take5_inputs,
    valid_patch_mask,
)


def test_extract_centered_patch_matches_matlab_index_extent():
    field = np.arange(12 * 12).reshape(12, 12, 1)
    patch = extract_centered_patches(field, np.array([5]), np.array([6]))
    assert patch.shape == (8, 8, 1)
    assert np.array_equal(patch[:, :, 0], field[2:10, 3:11, 0])


def test_take5_to_final_checkpoint_layout_removes_nhf_only():
    spatial = {name: np.full((8, 8, 2), index) for index, name in enumerate(TAKE5_SPATIAL_ORDER)}
    take5 = stack_take5_inputs(
        spatial,
        sst=np.array([10.0, 11.0]),
        sss=np.array([33.0, 34.0]),
        day_of_year=np.array([100, 101]),
        eddy_signal=np.array([0, 1]),
    )
    final = remove_nhf_for_final_checkpoint(take5)
    assert take5.shape == (8, 8, 14, 2)
    assert final.shape == (8, 8, 13, 2)
    assert np.all(final[:, :, 6, :] == take5[:, :, 7, :])
    assert np.all(final[:, :, 12, :] == take5[:, :, 13, :])


def test_missing_data_and_ow_rules():
    inputs = np.ones((8, 8, 13, 2))
    inputs[:5, :, :, 1] = np.nan
    assert valid_patch_mask(inputs).tolist() == [True, False]
    assert ow_eddy_mask(np.array([-0.3, -0.1]), np.ones(2)).tolist() == [True, False]
