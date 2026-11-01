"""Warping images by a flow and resizing flows."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ._ops import check_flow, check_image, check_size, pad_border

_WARP_MODES = ("bilinear", "nearest", "bicubic")
_PADDING_MODES = ("border", "zeros", "reflection")
_RESIZE_MODES = ("bilinear", "nearest", "bicubic", "area")


def _normalize(coord: Tensor, size: int) -> Tensor:
    """Pixel coordinates to grid_sample's [-1, 1] range, with ``align_corners=False``."""
    return (2 * coord + 1) / size - 1


def _reflect(coord: Tensor, size: int) -> Tensor:
    """Mirror pixel coordinates about the centres of the edge pixels into ``[0, size - 1]``.

    Replaces grid_sample's reflection, which torch.compile on CUDA maps from the far edge to 0
    for some sizes. This triangle wave is continuous, so rounding in the remainder is harmless.
    """
    if size == 1:
        return coord * 0
    last = size - 1
    return last - (torch.remainder(coord, 2 * last) - last).abs()


def warp(
    image: Tensor,
    flow: Tensor,
    *,
    mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
    padding_mode: Literal["border", "zeros", "reflection"] = "border",
) -> Tensor:
    """Sample ``image`` at the positions displaced by ``flow``.

    ``out[..., y, x] = image[..., y + flow_y[y, x], x + flow_x[y, x]]``, so
    ``warp(frame1, flow) ≈ frame0`` for a flow estimated from ``frame0`` to ``frame1``.
    Pixels whose flow is not finite are NaN, with zero gradient. float16 and bfloat16 inputs
    are sampled in float32.

    Parameters
    ----------
    image : Tensor
        Images, shape ``(B, C, H, W)``.
    flow : Tensor
        Displacements in pixels, shape ``(B, 2, H, W)``; channel 0 is x, channel 1 is y.
    mode : {"bilinear", "nearest", "bicubic"}
        Interpolation, as in :func:`torch.nn.functional.grid_sample`.
    padding_mode : {"border", "zeros", "reflection"}
        Values used for positions outside the image, as in :func:`torch.nn.functional.grid_sample`;
        ``"reflection"`` mirrors about the centres of the edge pixels.

    Returns
    -------
    Tensor
        Warped images, shape ``(B, C, H, W)``, with the dtype of ``image``.
    """
    check_image(image)
    check_flow(flow, image, like_name="image")
    if mode not in _WARP_MODES:
        raise ValueError(f"mode must be one of {_WARP_MODES}, got {mode!r}")
    if padding_mode not in _PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}")
    dtype = image.dtype
    compute = torch.promote_types(dtype, torch.float32)  # half-precision grids cannot address pixels
    image, flow = image.to(compute), flow.to(compute)
    _, _, height, width = image.shape
    # grid_sample's CPU backward crashes on NaN positions. Masking with `bad` itself instead of
    # `finite` miscompiles (torch 2.15, CUDA, dynamic shapes).
    finite = torch.isfinite(flow)
    bad = ~finite.all(dim=1, keepdim=True)
    flow = torch.where(finite, flow, 0.0)
    x = torch.arange(width, dtype=compute, device=flow.device) + flow[:, 0]
    y = torch.arange(height, dtype=compute, device=flow.device).unsqueeze(-1) + flow[:, 1]
    if padding_mode == "reflection":
        x, y = _reflect(x, width), _reflect(y, height)
        if mode == "bicubic":  # taps reach 2 pixels out; bicubic of the mirrored image is symmetric too
            image = pad_border(image, (2, 2, 2, 2), "reflect101")
            x, y, height, width = x + 2, y + 2, height + 4, width + 4
        padding_mode = "border"
    else:
        # beyond these bounds every tap is in the padding: same values and gradients, finite grid
        x, y = x.clamp(-3, width + 2), y.clamp(-3, height + 2)
    grid = torch.stack((_normalize(x, width), _normalize(y, height)), dim=-1)
    out = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return torch.where(bad, torch.nan, out).to(dtype)


def resize_flow(
    flow: Tensor,
    size: tuple[int, int],
    *,
    mode: Literal["bilinear", "nearest", "bicubic", "area"] = "bilinear",
) -> Tensor:
    """Resize a flow to ``size`` and rescale its vectors to the new pixel size.

    Parameters
    ----------
    flow : Tensor
        Displacements in pixels, shape ``(B, 2, H, W)``.
    size : tuple of int
        Output ``(height, width)``.
    mode : {"bilinear", "nearest", "bicubic", "area"}
        Interpolation, as in :func:`torch.nn.functional.interpolate` with ``align_corners=False``;
        ``"nearest"`` is its ``"nearest-exact"``.

    Returns
    -------
    Tensor
        Flow of shape ``(B, 2, height, width)`` in pixels of the new grid.
    """
    check_flow(flow)
    height, width = check_size(size)
    if mode not in _RESIZE_MODES:
        raise ValueError(f"mode must be one of {_RESIZE_MODES}, got {mode!r}")
    align_corners = False if mode in ("bilinear", "bicubic") else None
    interpolation = "nearest-exact" if mode == "nearest" else mode  # legacy "nearest" is off-centre
    out = F.interpolate(flow, size=(height, width), mode=interpolation, align_corners=align_corners)
    return torch.cat((out[:, :1] * (width / flow.shape[-1]), out[:, 1:] * (height / flow.shape[-2])), dim=1)
