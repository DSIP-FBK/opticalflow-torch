"""Colour coding of flows for plots."""

from __future__ import annotations

import torch
from torch import Tensor

from ._ops import as_scalar, check_flow
from .metrics import _length


def _colour_wheel() -> tuple[tuple[float, float, float], ...]:
    """The 55 colours of the Middlebury wheel (Baker et al., 2011), in [0, 1], from red through yellow,
    green, cyan, blue and magenta."""
    # (length, channel that ramps, rising, the other two channels) per segment RY, YG, GC, CB, BM, MR
    segments = (
        (15, 1, True, {0: 255, 2: 0}),
        (6, 0, False, {1: 255, 2: 0}),
        (4, 2, True, {0: 0, 1: 255}),
        (11, 1, False, {0: 0, 2: 255}),
        (13, 0, True, {1: 0, 2: 255}),
        (6, 2, False, {0: 255, 1: 0}),
    )
    colours = []
    for length, channel, rising, fixed in segments:
        for i in range(length):
            ramp = 255 * i // length
            colour = {**fixed, channel: ramp if rising else 255 - ramp}
            colours.append(tuple(colour[c] / 255 for c in range(3)))
    return tuple(colours)


_WHEEL = _colour_wheel()


def flow_to_rgb(flow: Tensor, *, max_magnitude: float | Tensor | None = None) -> Tensor:
    """Colour-code a flow with the standard optical flow colour wheel (Baker et al., 2011).

    Hue gives the direction and saturation the magnitude: white for no motion, the wheel
    colour at ``max_magnitude`` (red for +x, yellow for +y, cyan for -x, violet for -y).
    Larger magnitudes darken towards black (the colour divided by the magnitude relative to
    ``max_magnitude``, as flowpy does). The 55 Middlebury colours are spread evenly over the
    full circle, so the colours change continuously; the Middlebury code and torchvision's
    ``flow_to_image`` spread them over 54 steps, which leaves a jump at +x, and round to 8 bits.
    Pixels whose flow is not finite are NaN, with zero gradient.

    Parameters
    ----------
    flow : Tensor
        Flow, shape ``(B, 2, H, W)``.
    max_magnitude : float or Tensor, optional
        Magnitude shown at full saturation, positive, by default the largest finite magnitude in
        each flow.

    Returns
    -------
    Tensor
        RGB images in [0, 1], shape ``(B, 3, H, W)``.
    """
    check_flow(flow)
    finite = torch.isfinite(flow).all(dim=1)
    flow = torch.where(finite.unsqueeze(1), flow, 0.0)  # so that no NaN reaches the gradient of the maximum
    length = _length(flow)
    if max_magnitude is None:
        scale = length.amax(dim=(-2, -1), keepdim=True)
    else:
        scale = as_scalar(max_magnitude, flow, "max_magnitude")
        if not isinstance(max_magnitude, Tensor) and not max_magnitude > 0:
            raise ValueError(f"max_magnitude must be positive, got {max_magnitude}")
    radius = (length / scale.clamp_min(torch.finfo(flow.dtype).tiny)).unsqueeze(-1)
    # atan2 has no gradient at the origin, where the colour is white whatever the angle; elsewhere
    # it takes the unit vector, as its backward divides by u² + v², which overflows for tiny flows
    still = length == 0
    norm = torch.where(still, 1.0, length)
    u = torch.where(still, 1.0, flow[:, 0] / norm)
    v = torch.where(still, 0.0, flow[:, 1] / norm)
    wheel = torch.tensor(_WHEEL, dtype=flow.dtype, device=flow.device)
    count = wheel.shape[0]
    # Middlebury's angle convention: position 0 (red) is +x, then +y (down) through yellow, and
    # position `count` is +x again
    position = (torch.atan2(-v, -u) / torch.pi + 1) / 2 * count
    index0 = position.floor().long().clamp(0, count - 1)
    index1 = (index0 + 1) % count
    frac = (position - index0).unsqueeze(-1)
    colour = (1 - frac) * wheel[index0] + frac * wheel[index1]  # (B, H, W, 3)
    # a where rather than clamps of the radius, which would both pass no gradient at radius 1
    rgb = torch.where(radius <= 1, 1 - radius * (1 - colour), colour / radius.clamp(min=1))
    rgb = torch.where(finite.unsqueeze(-1), rgb, torch.nan)
    return rgb.permute(0, 3, 1, 2).contiguous()
