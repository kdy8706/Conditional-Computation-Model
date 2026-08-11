import numpy as np

from ocean_ccm.metrics import profile_metrics


def test_perfect_prediction():
    observed = np.arange(24, dtype=float).reshape(4, 3, 2)
    result = profile_metrics(observed, observed.copy())
    assert np.allclose(result["rmse"], 0)
    assert np.allclose(result["mae"], 0)
    assert np.allclose(result["r2"], 1)
    assert np.allclose(result["nrmse"], 0)
