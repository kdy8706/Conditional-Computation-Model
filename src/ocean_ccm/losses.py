"""Loss functions used by the legacy and paper-reported training profiles."""

from __future__ import annotations

import torch
from torch import Tensor


def _valid_mean(loss: Tensor, valid: Tensor) -> Tensor:
    count = valid.sum()
    if count.item() == 0:
        return loss.sum() * 0.0
    return (loss * valid).sum() / count


def masked_huber_loss(
    output: Tensor,
    target: Tensor,
    *,
    delta: float = 1.0,
    missing_value: float = -999.0,
) -> Tensor:
    """Huber loss that ignores targets equal to ``missing_value``."""

    valid = target.ne(missing_value).to(output.dtype)
    error = target - output
    absolute_error = error.abs()
    quadratic = torch.where(
        absolute_error < delta,
        0.5 * error.square(),
        delta * absolute_error - 0.5 * delta**2,
    )
    return _valid_mean(quadratic, valid)


def focal_masked_huber_loss(
    output: Tensor,
    target: Tensor,
    *,
    delta: float = 1.0,
    gamma: float = 1.5,
    missing_value: float = -999.0,
) -> Tensor:
    """Legacy focal reweighting applied to masked Huber loss."""

    valid = target.ne(missing_value).to(output.dtype)
    error = target - output
    absolute_error = error.abs()
    quadratic = torch.where(
        absolute_error < delta,
        0.5 * error.square(),
        delta * absolute_error - 0.5 * delta**2,
    )
    weight = (1.0 + absolute_error / delta).pow(gamma)
    return _valid_mean(quadratic * weight, valid)


def branch_losses(
    profile: str,
    *,
    non_eddy_gamma: float = 1.5,
    eddy_gamma: float = 2.0,
):
    """Return loss callables for the two documented training profiles."""

    if profile == "legacy_try3":
        return (
            lambda output, target: focal_masked_huber_loss(
                output, target, gamma=non_eddy_gamma
            ),
            masked_huber_loss,
        )
    if profile == "legacy_try2":
        return (
            masked_huber_loss,
            lambda output, target: focal_masked_huber_loss(
                output, target, gamma=non_eddy_gamma
            ),
        )
    if profile == "paper_focal":
        return (
            lambda output, target: focal_masked_huber_loss(
                output, target, gamma=non_eddy_gamma
            ),
            lambda output, target: focal_masked_huber_loss(
                output, target, gamma=eddy_gamma
            ),
        )
    raise ValueError(f"Unknown loss profile: {profile}")
