"""Flow accuracy and reliability measures."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ._ops import as_scalar, check_flow, check_frames
from .warp import warp


def _length(vectors: Tensor) -> Tensor:
    """Length of ``vectors`` along dimension 1, with all derivatives 0 at 0.

    ``torch.linalg.vector_norm`` has a zero gradient at 0 but a NaN second derivative.
    """
    square = vectors.square().sum(dim=1)
    nonzero = square != 0  # true for NaN, which must stay NaN
    return torch.where(nonzero, torch.where(nonzero, square, 1).sqrt(), 0)


def endpoint_error(flow: Tensor, reference: Tensor) -> Tensor:
    """Length of the difference between two flows, per pixel.

    Where the flows are equal the gradient is 0 (a subgradient of the length), and so are the
    higher derivatives.

    Parameters
    ----------
    flow, reference : Tensor
        Flows, shape ``(B, 2, H, W)``.

    Returns
    -------
    Tensor
        Endpoint error in pixels, shape ``(B, H, W)``.
    """
    check_flow(flow)
    check_flow(reference, flow, name="reference", like_name="flow")
    return _length(flow - reference)


def photometric_error(
    frame0: Tensor,
    frame1: Tensor,
    flow: Tensor,
    *,
    padding_mode: Literal["border", "zeros", "reflection"] = "border",
) -> Tensor:
    """Absolute difference between ``frame0`` and ``frame1`` warped back by ``flow``.

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``.
    flow : Tensor
        Flow from ``frame0`` to ``frame1``, shape ``(B, 2, H, W)``.
    padding_mode : {"border", "zeros", "reflection"}
        Passed to :func:`~opticalflow_torch.warp`.

    Returns
    -------
    Tensor
        ``|warp(frame1, flow) - frame0|``, shape ``(B, C, H, W)``.
    """
    check_frames(frame0, frame1)
    check_flow(flow, frame0)
    return (warp(frame1, flow, padding_mode=padding_mode) - frame0).abs()


def consistency(
    forward_flow: Tensor,
    backward_flow: Tensor,
    *,
    alpha: float | Tensor = 0.01,
    beta: float | Tensor = 0.5,
) -> Tensor:
    """Soft agreement between a forward flow and the backward flow, per pixel.

    Following a pixel forward and then back should return it to its start. With
    ``b = warp(backward_flow, forward_flow)`` the result is
    ``exp(-|f + b|² / (alpha * (|f|² + |b|²) + beta))``: 1 where the two flows agree,
    towards 0 where they do not, typically at occlusions and where rain appears or
    disappears. ``alpha`` and ``beta`` are those of the usual hard check (Sundaram et al.,
    2010).

    Parameters
    ----------
    forward_flow : Tensor
        Flow from ``frame0`` to ``frame1``, shape ``(B, 2, H, W)``.
    backward_flow : Tensor
        Flow from ``frame1`` to ``frame0``, shape ``(B, 2, H, W)``.
    alpha, beta : float or Tensor
        Tolerance relative to the flow magnitudes and absolute tolerance (pixels²); ``alpha``
        non-negative and ``beta`` positive.

    Returns
    -------
    Tensor
        Agreement in [0, 1], shape ``(B, H, W)``; large mismatches underflow to exactly 0.
    """
    check_flow(forward_flow, name="forward_flow")
    check_flow(backward_flow, forward_flow, name="backward_flow", like_name="forward_flow")
    relative, absolute = as_scalar(alpha, forward_flow, "alpha"), as_scalar(beta, forward_flow, "beta")
    if not isinstance(alpha, Tensor) and not alpha >= 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if not isinstance(beta, Tensor) and not beta > 0:
        raise ValueError(f"beta must be positive, got {beta}")
    back = warp(backward_flow, forward_flow)
    mismatch = (forward_flow + back).square().sum(dim=1)
    tolerance = relative * (forward_flow.square().sum(dim=1) + back.square().sum(dim=1)) + absolute
    return torch.exp(-mismatch / tolerance)
