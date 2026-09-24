"""Exact-value, factorized-gradient reconstruction objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .controller import FactorDecision

try:
    from fused_ssim import fused_ssim as _fused_ssim
    from fused_ssim_cuda import fusedssim, fusedssim_backward

    _FUSED_BACKEND_AVAILABLE = True
except ImportError:
    _fused_ssim = None
    fusedssim = None
    fusedssim_backward = None
    _FUSED_BACKEND_AVAILABLE = False


_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def fused_backend_available() -> bool:
    return _FUSED_BACKEND_AVAILABLE


class _ExactValueSelectedChannelSSIM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        image,
        target,
        selected_channel,
        temporal_weight,
        importance_correction,
    ):
        if not _FUSED_BACKEND_AVAILABLE:
            raise RuntimeError("FactorGrad-GS requires the fused-ssim CUDA backend")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("FactorGrad-GS expects an NCHW RGB image")
        if image.shape != target.shape:
            raise ValueError("image and target must have identical shapes")
        channel = int(selected_channel)
        if channel not in (0, 1, 2):
            raise ValueError("selected_channel must be 0, 1, or 2")

        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(
            _C1, _C2, image, target, True
        )
        ctx.save_for_backward(
            image.detach(),
            target.detach(),
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        )
        ctx.selected_channel = channel
        ctx.temporal_weight = float(temporal_weight)
        ctx.importance_correction = bool(importance_correction)
        return ssim_map.mean()

    @staticmethod
    def backward(ctx, output_gradient):
        image, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        channel_slice = slice(ctx.selected_channel, ctx.selected_channel + 1)
        selected_image = image[:, channel_slice].contiguous()
        selected_target = target[:, channel_slice].contiguous()

        # The corrected estimator divides by CH=1 elements, supplying the
        # Monte Carlo channel factor 3. temporal_weight supplies 1 / q_t.
        # The uncorrected ablation instead uses the RGB element count.
        denominator = (
            selected_image.numel()
            if ctx.importance_correction
            else image.numel()
        )
        map_gradient = torch.ones_like(selected_image) * (
            output_gradient * ctx.temporal_weight / denominator
        )
        selected_gradient = fusedssim_backward(
            _C1,
            _C2,
            selected_image,
            selected_target,
            map_gradient.contiguous(),
            dm_dmu1[:, channel_slice].contiguous(),
            dm_dsigma1_sq[:, channel_slice].contiguous(),
            dm_dsigma12[:, channel_slice].contiguous(),
        )
        image_gradient = torch.zeros_like(image)
        image_gradient[:, channel_slice] = selected_gradient
        return image_gradient, None, None, None, None


@dataclass(frozen=True)
class FactorLoss:
    objective: torch.Tensor
    exact_loss: torch.Tensor
    l1: torch.Tensor
    exact_dssim: torch.Tensor
    decision: FactorDecision


def _fused(image: torch.Tensor, target: torch.Tensor, train: bool) -> torch.Tensor:
    if not _FUSED_BACKEND_AVAILABLE:
        raise RuntimeError("FactorGrad-GS requires the fused-ssim CUDA backend")
    return _fused_ssim(image, target, train=train)


def factorgrad_loss(
    image: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float,
    decision: FactorDecision,
    importance_correction: bool = True,
) -> FactorLoss:
    """Compute an exact scalar with the structural derivative chosen by ``decision``."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a CHW RGB tensor")
    if image.shape != target.shape:
        raise ValueError("image and target must have identical shapes")

    l1 = torch.abs(image - target).mean()
    batched_image = image.unsqueeze(0)
    batched_target = target.unsqueeze(0)

    if decision.mode in ("baseline_full", "full_rgb"):
        structural_ssim = _fused(batched_image, batched_target, train=True)
        exact_dssim = 1.0 - structural_ssim
        objective_dssim = exact_dssim
    elif decision.mode == "reweighted_full_rgb":
        structural_ssim = _fused(batched_image, batched_target, train=True)
        exact_dssim = 1.0 - structural_ssim
        temporal_weight = (
            decision.temporal_weight if importance_correction else 1.0
        )
        objective_dssim = exact_dssim.detach() + temporal_weight * (
            exact_dssim - exact_dssim.detach()
        )
    elif decision.mode == "factorized_channel":
        structural_ssim = _ExactValueSelectedChannelSSIM.apply(
            batched_image,
            batched_target,
            decision.selected_channel,
            decision.temporal_weight if importance_correction else 1.0,
            importance_correction,
        )
        exact_dssim = 1.0 - structural_ssim
        objective_dssim = exact_dssim
    elif decision.mode == "l1_only":
        with torch.no_grad():
            exact_dssim = 1.0 - _fused(
                batched_image.detach(), batched_target, train=False
            )
        objective_dssim = exact_dssim.detach()
    else:
        raise ValueError("unsupported FactorGrad mode: {}".format(decision.mode))

    objective = (1.0 - lambda_dssim) * l1 + lambda_dssim * objective_dssim
    exact_loss = (1.0 - lambda_dssim) * l1.detach() + lambda_dssim * exact_dssim.detach()
    return FactorLoss(
        objective=objective,
        exact_loss=exact_loss,
        l1=l1,
        exact_dssim=exact_dssim.detach(),
        decision=decision,
    )
