import pytest

torch = pytest.importorskip("torch")

from ocean_ccm.model import ConditionalModel, collect_outputs


def test_forward_restores_original_batch_order():
    model = ConditionalModel()
    model.eval()
    spatial = torch.randn(4, 9, 8, 8)
    vector = torch.randn(4, 3)
    pressure = torch.randn(4, 13)
    signal = torch.zeros(4, 8, 8)
    signal[1, :, :] = 1
    signal[3, :, :] = 1

    with torch.no_grad():
        grouped = model(spatial, vector, pressure, signal)
        output = collect_outputs(grouped, batch_size=4)

    assert set(grouped) == {"normal", "outlier"}
    assert output.shape == (4, 13, 2)


def test_legacy_parameter_count_and_keys():
    model = ConditionalModel()
    assert sum(parameter.numel() for parameter in model.parameters()) == 556_324
    keys = model.state_dict().keys()
    assert "normal_gated_conv1.conv_gate.0.weight" in keys
    assert "outlier_expand_fc.12.weight" in keys


def test_pool_mode_is_explicit_per_model_profile():
    assert ConditionalModel(pool_mode="max").pool_mode == "max"
    assert ConditionalModel(pool_mode="avg").pool_mode == "avg"
