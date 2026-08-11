"""Legacy-checkpoint-compatible conditional computation model."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


MaskMode = Literal["legacy", "zero_invalid"]
PoolMode = Literal["max", "avg"]


class GatedConv2D(nn.Module):
    """Feature convolution multiplied by a learned tanh gate.

    ``legacy`` reproduces the supplied implementation. ``zero_invalid`` is an
    experimental correction that zeros sentinel values before convolution and
    requires explicit retraining/validation before scientific use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: str | int = "valid",
        residual: Tensor | None = None,
        *,
        missing_value: float = -999.0,
        mask_mode: MaskMode = "legacy",
    ) -> None:
        super().__init__()
        if padding == "valid":
            convolution_padding: str | int = 0
        elif padding == "same":
            convolution_padding = "same"
        else:
            convolution_padding = padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.residual = residual
        self.missing_value = missing_value
        self.mask_mode = mask_mode

        self.conv_feature = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=convolution_padding,
        )
        self.conv_gate = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=convolution_padding,
            ),
            nn.Tanh(),
        )

        nn.init.kaiming_normal_(self.conv_feature.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_gate[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.conv_feature.bias)
        nn.init.zeros_(self.conv_gate[0].bias)

    def forward(self, x: Tensor) -> Tensor:
        channel_valid = x.ne(self.missing_value)
        if self.mask_mode == "legacy":
            spatial_valid = channel_valid.any(dim=1, keepdim=True)
            convolution_input = x
        elif self.mask_mode == "zero_invalid":
            spatial_valid = channel_valid.all(dim=1, keepdim=True)
            convolution_input = torch.where(channel_valid, x, torch.zeros_like(x))
        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")

        feature = F.gelu(self.conv_feature(convolution_input))
        gate = self.conv_gate(convolution_input)
        resized = F.interpolate(
            spatial_valid.to(feature.dtype),
            size=feature.shape[2:],
            mode="nearest",
        )
        output = feature * gate * resized
        if self.residual is not None:
            output = output + self.residual
        return output


class ConditionalModel(nn.Module):
    """Hard-routed eddy/non-eddy experts with sequential depth regression.

    Attribute names intentionally match the legacy ``state_dict``. In legacy
    terminology, ``normal`` means non-eddy and ``outlier`` means eddy.
    """

    def __init__(
        self,
        input_shape_spatial: tuple[int, int, int] = (9, 8, 8),
        input_shape_dense: int = 3,
        input_shape_pressure: int = 14,
        input_eddy_sig: tuple[int, int] = (8, 8),
        num_depths: int = 13,
        drop: float = 0.2,
        *,
        mask_mode: MaskMode = "legacy",
        pool_mode: PoolMode = "max",
        decision_index: tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        del input_shape_pressure, input_eddy_sig
        self.num_depths = num_depths
        self.drop = drop
        if pool_mode not in ("max", "avg"):
            raise ValueError(f"Unknown pool mode: {pool_mode!r}")
        self.pool_mode = pool_mode
        self.decision_index = decision_index

        self._build_expert(
            "normal", input_shape_spatial[0], input_shape_dense, drop, mask_mode
        )
        self._build_expert(
            "outlier", input_shape_spatial[0], input_shape_dense, drop, mask_mode
        )
        self.dropout = nn.Dropout(drop)

    def _build_expert(
        self,
        prefix: str,
        spatial_channels: int,
        dense_features: int,
        drop: float,
        mask_mode: MaskMode,
    ) -> None:
        setattr(
            self,
            f"{prefix}_gated_conv1",
            GatedConv2D(
                spatial_channels,
                20,
                (3, 3),
                stride=1,
                padding=0,
                mask_mode=mask_mode,
            ),
        )
        setattr(self, f"{prefix}_gated_conv2", nn.Conv2d(20, 40, (3, 3)))
        setattr(self, f"{prefix}_gated_conv3", nn.Conv2d(40, 80, (3, 3)))
        setattr(self, f"{prefix}_max_pool", nn.MaxPool2d(2, 2))
        setattr(self, f"{prefix}_avg_pool", nn.AvgPool2d(2, 2))
        setattr(self, f"{prefix}_flatten", nn.Flatten())
        setattr(self, f"{prefix}_layer_norm_spatial", nn.BatchNorm1d(80))

        setattr(self, f"{prefix}_dense_fc1", nn.Linear(dense_features, 6))
        setattr(self, f"{prefix}_dense_fc2", nn.Linear(6, 12))
        setattr(self, f"{prefix}_dense_fc3", nn.Linear(12, 24))
        setattr(self, f"{prefix}_layer_norm_dense", nn.BatchNorm1d(24))

        residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3, 6),
                    nn.GELU(),
                    nn.Linear(6, 12),
                    nn.GELU(),
                    nn.Linear(12, 24),
                    nn.GELU(),
                    nn.BatchNorm1d(24),
                )
                for _ in range(self.num_depths)
            ]
        )
        setattr(self, f"{prefix}_residual_fc", residual_layers)

        # The legacy state dictionary contains one expansion layer per depth,
        # although only index zero is reachable. Retain all layers to load it.
        setattr(
            self,
            f"{prefix}_expand_fc",
            nn.ModuleList([nn.Linear(104, 128) for _ in range(self.num_depths)]),
        )
        setattr(
            self,
            f"{prefix}_depth_fc",
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(128, 32),
                        nn.GELU(),
                        nn.Dropout(drop),
                        nn.Linear(32, 8),
                        nn.GELU(),
                        nn.Dropout(drop),
                        nn.Linear(8, 2),
                    )
                    for _ in range(self.num_depths)
                ]
            ),
        )

    def _run_expert(
        self,
        prefix: str,
        cnn_input: Tensor,
        dense_input: Tensor,
        pressure_input: Tensor,
        indices: Tensor,
    ) -> Tensor:
        x = cnn_input[indices]
        dense = dense_input[indices]
        pressure = pressure_input[indices]

        x = getattr(self, f"{prefix}_gated_conv1")(x)
        x = getattr(self, f"{prefix}_gated_conv2")(x)
        x = getattr(self, f"{prefix}_gated_conv3")(x)
        x = getattr(self, f"{prefix}_{self.pool_mode}_pool")(x)
        spatial_features = getattr(self, f"{prefix}_flatten")(x)
        spatial_features = getattr(self, f"{prefix}_layer_norm_spatial")(
            spatial_features
        )

        dense = F.gelu(getattr(self, f"{prefix}_dense_fc1")(dense))
        dense = F.gelu(getattr(self, f"{prefix}_dense_fc2")(dense))
        dense = F.gelu(getattr(self, f"{prefix}_dense_fc3")(dense))
        dense_features = getattr(self, f"{prefix}_layer_norm_dense")(dense)
        shared_features = torch.cat([spatial_features, dense_features], dim=-1)

        residual_layers = getattr(self, f"{prefix}_residual_fc")
        expansion_layers = getattr(self, f"{prefix}_expand_fc")
        output_layers = getattr(self, f"{prefix}_depth_fc")
        predictions: list[Tensor] = []
        previous_output: Tensor | None = None

        for depth in range(self.num_depths):
            if previous_output is None:
                depth_input = expansion_layers[depth](shared_features)
            else:
                previous_pressure = pressure[:, depth].unsqueeze(-1)
                residual = residual_layers[depth](
                    torch.cat([previous_output, previous_pressure], dim=-1)
                )
                depth_input = torch.cat([shared_features, residual], dim=-1)

            previous_output = output_layers[depth](depth_input)
            predictions.append(previous_output.unsqueeze(1))

        return torch.cat(predictions, dim=1)

    def forward(
        self,
        cnn_input: Tensor,
        dense_input: Tensor,
        pressure_input: Tensor,
        eddy_sig: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        row, column = self.decision_index
        is_eddy = eddy_sig[:, row, column] > 0
        normal_indices = torch.nonzero(~is_eddy, as_tuple=True)[0]
        outlier_indices = torch.nonzero(is_eddy, as_tuple=True)[0]

        outputs: dict[str, dict[str, Tensor]] = {}
        if normal_indices.numel() > 0:
            outputs["normal"] = {
                "indices": normal_indices,
                "outputs": self._run_expert(
                    "normal", cnn_input, dense_input, pressure_input, normal_indices
                ),
            }
        if outlier_indices.numel() > 0:
            outputs["outlier"] = {
                "indices": outlier_indices,
                "outputs": self._run_expert(
                    "outlier", cnn_input, dense_input, pressure_input, outlier_indices
                ),
            }
        return outputs


def collect_outputs(
    outputs_dict: dict[str, dict[str, Tensor]],
    batch_size: int,
    num_depths: int = 13,
    num_features: int = 2,
    *,
    reference: Tensor | None = None,
) -> Tensor:
    """Restore expert-grouped outputs to original batch order."""

    if reference is None:
        if not outputs_dict:
            raise ValueError("reference is required when outputs_dict is empty")
        reference = next(iter(outputs_dict.values()))["outputs"]
    outputs = reference.new_zeros((batch_size, num_depths, num_features))
    for branch in ("normal", "outlier"):
        if branch in outputs_dict:
            outputs[outputs_dict[branch]["indices"]] = outputs_dict[branch]["outputs"]
    return outputs
