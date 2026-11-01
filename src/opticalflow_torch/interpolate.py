"""Motion-compensated interpolation between two frames."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ._ops import as_scalar, check_flow, check_frames, no_autocast
from .warp import warp

_SMOOTHING = 0.25  # half-width of the box smoothing the blend's distribution function, in mean target quantile spacings


def _times(t: float | Sequence[float] | Tensor, like: Tensor) -> Tensor:
    """``t`` as a 1-D tensor with the dtype and device of ``like``; Python numbers must be in [0, 1]."""
    if isinstance(t, Tensor):
        if t.dim() > 1:
            raise ValueError(f"t must be a scalar or a 1-D tensor, got shape {tuple(t.shape)}")
        times = t.reshape(-1).to(dtype=like.dtype, device=like.device)
    else:
        values = [t] if isinstance(t, (int, float)) else t
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise TypeError(f"t must be a float, a sequence of floats or a tensor, got {type(t).__name__}")
        for value in values:
            if isinstance(value, (int, float)) and not 0 <= value <= 1:
                raise ValueError(f"t must be in [0, 1], got {value}")
        if all(isinstance(value, (int, float)) for value in values):
            times = torch.tensor([float(value) for value in values], dtype=like.dtype, device=like.device)
        else:
            times = torch.stack([as_scalar(value, like, "t") for value in values]) if values else like.new_empty(0)
    if times.numel() == 0:
        raise ValueError("t must contain at least one time")
    return times


def _quantile_taps(n: int, quantiles: int, like: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Order statistics and weights of the quantiles at probabilities ``k / (quantiles - 1)`` of ``n`` values.

    The linear interpolation of :func:`torch.quantile`, with exact integer positions ``k (n - 1) / (quantiles - 1)``.
    """
    position = torch.arange(quantiles, device=like.device) * (n - 1)
    low = position // (quantiles - 1)
    frac = ((position - low * (quantiles - 1)) / (quantiles - 1)).to(like.dtype)
    return low, (low + 1).clamp(max=n - 1), frac


def _quantiles(x: Tensor, taps: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Quantiles of ``x`` over its last two dimensions, shape ``(..., quantiles)``, by sorting."""
    low, high, frac = taps
    ordered = x.flatten(-2).sort(dim=-1).values
    below, above = ordered.index_select(-1, low), ordered.index_select(-1, high)
    return below + frac * (above - below)  # equal to `below` when the two are tied


def _cumsum(terms: Tensor) -> Tensor:
    """Cumulative sums of non-negative ``terms`` along the last dimension, in two parts that keep differences exact.

    A plain cumsum rounds every sum to the precision of the total. Multiples of a power of two near the total add
    exactly, so only the cumsum of the remainders below it rounds; it also carries the gradient. Shape ``(2, ...)``.
    """
    bits = round(-math.log2(torch.finfo(terms.dtype).eps)) - 12
    total = terms.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(terms.dtype).tiny)
    quantum = torch.exp2(total.log2().floor() - bits)
    coarse = (terms / quantum).round() * quantum
    return torch.stack((coarse.cumsum(dim=-1), (terms - coarse).cumsum(dim=-1)))


def _distribution(position: Tensor, knots: Tensor) -> tuple[Tensor, Tensor]:
    """Index of the last knot at or below ``position``, and the blend's distribution function there.

    The function is piecewise linear through ``knots``, the quantiles at probabilities ``k / (Q - 1)``, and 1 beyond
    the last; ``position`` is at least the first knot.
    """
    steps = knots.shape[-1] - 1
    index = (torch.searchsorted(knots, position, side="right") - 1).clamp(min=0)
    inner = index < steps  # then the next knot is above `position`, so the gap is positive
    x0 = knots.gather(-1, index)
    gap = torch.where(inner, knots.gather(-1, (index + 1).clamp(max=steps)) - x0, 1.0)
    return index, (index + torch.where(inner, (position - x0) / gap, 0.0)) / steps


def _window_integral(start: Tensor, end: Tensor, width: Tensor, knots: Tensor, area: Tensor) -> Tensor:
    """Integral of the blend's distribution function over ``[start, end]``, of length ``width``.

    ``area`` holds the integrals up to the knots, from :func:`_cumsum`. Taken as ``width`` times the value at
    ``start`` plus the integral of the increase from there, so that rounding of ``start`` and ``end`` costs only the
    increase.
    """
    steps = knots.shape[-1] - 1
    first, at_start = _distribution(start, knots)
    last, at_end = _distribution(end, knots)
    after = (first + 1).clamp(max=steps)
    x_after, x_last = knots.gather(-1, after), knots.gather(-1, last)
    p_after, p_last = after.to(knots.dtype) / steps, last.to(knots.dtype) / steps
    # from `start` to the next knot, the knot intervals in between, and from the last knot to `end`
    head = (x_after - start) * (p_after - at_start) / 2
    coarse, fine = area
    between = coarse.gather(-1, last) - coarse.gather(-1, after) + (fine.gather(-1, last) - fine.gather(-1, after))
    between = between - at_start * (x_last - x_after)
    tail = (end - x_last) * (p_last + at_end - 2 * at_start) / 2
    rise = torch.where(first == last, width * (at_end - at_start) / 2, head + between + tail)
    return width * at_start + rise


def _match_distribution(blend: Tensor, frame0: Tensor, frame1: Tensor, t: Tensor, quantiles: int) -> Tensor:
    """Map ``blend`` (B, T, C, H, W) to the distribution interpolated between the frames at times ``t``."""
    # double precision (single on MPS): the smoothed distribution function divides integrals by a narrow window
    work = torch.float32 if blend.device.type == "mps" else torch.float64
    steps = quantiles - 1
    taps = _quantile_taps(blend.shape[-2] * blend.shape[-1], quantiles, blend)
    t = t.view(1, -1, 1, 1)
    target = (1 - t) * _quantiles(frame0, taps).unsqueeze(1) + t * _quantiles(frame1, taps).unsqueeze(1)
    target = target.to(work)
    # the knots must be non-decreasing for searchsorted; the interpolation could break that by an ulp
    knots = _quantiles(blend, taps).to(work).cummax(dim=-1).values
    offset = blend.flatten(-2).to(work) - knots[..., :1]
    knots = knots - knots[..., :1]
    middle = (torch.arange(steps, device=blend.device, dtype=work) + 0.5) / steps  # mean probability per interval
    area = torch.nn.functional.pad(_cumsum(knots.diff(dim=-1) * middle), (1, 0))
    # The distribution function jumps where knots tie (dry areas, saturated values), so mapping values through it
    # would jump as a tie breaks. Averaged over [v - half, v + half] it is continuous; extended below the minimum
    # as an odd function, it keeps the minimum at probability 0, so that the dry pixels of a blend drier than the
    # target (rain lost at the border) stay dry. Scaling `half` with the target's range rather than the blend's
    # also keeps the map continuous as the blend becomes constant.
    half = (_SMOOTHING / steps * (target[..., -1:] - target[..., :1])).clamp(min=torch.finfo(work).tiny ** 0.5)
    width = 2 * torch.minimum(offset, half)  # of the window folded at the minimum, [|offset - half|, offset + half]
    probability = _window_integral((offset - half).abs(), offset + half, width, knots, area) / (2 * half)
    # then through the target's quantile function, piecewise linear between the `target` knots
    position = probability.clamp(0, 1) * steps
    # NaN (from non-finite inputs) would index out of bounds; it stays NaN through `position`
    lower = position.detach().nan_to_num(0).floor().clamp(max=steps - 1)
    y0, y1 = target.gather(-1, lower.long()), target.gather(-1, lower.long() + 1)
    return (y0 + (position - lower) * (y1 - y0)).to(blend.dtype).view_as(blend)


@no_autocast
def interpolate_frames(
    frame0: Tensor,
    frame1: Tensor,
    flow: Tensor,
    t: float | Sequence[float] | Tensor,
    *,
    backward_flow: Tensor | None = None,
    match_distribution: bool = False,
    quantiles: int = 256,
) -> Tensor:
    """Frames at times ``t`` between ``frame0`` (t=0) and ``frame1`` (t=1).

    Each frame blends ``frame0`` moved forward by ``t * flow`` and ``frame1`` moved back by
    ``(1 - t) * flow``, weighted ``1 - t`` and ``t`` (the "advection correction" used for
    radar). With ``backward_flow``, ``frame1`` is moved with its own motion instead.
    Times 0 and 1 return ``frame0`` and ``frame1`` exactly, with the gradients of the blend.

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``.
    flow : Tensor
        Flow from ``frame0`` to ``frame1``, shape ``(B, 2, H, W)``.
    t : float, sequence of float or Tensor
        Times in [0, 1]; a tensor is a scalar or 1-D and may require grad.
    backward_flow : Tensor, optional
        Flow from ``frame1`` to ``frame0``, shape ``(B, 2, H, W)``.
    match_distribution : bool
        Map each blend's values so that their distribution is the one interpolated between
        ``frame0`` and ``frame1`` (probability matching), per image and channel: through the
        blend's distribution function, piecewise linear between ``quantiles`` quantiles and
        smoothed over a quarter of the mean spacing of the target quantiles, then through the
        target's quantile function. The map is continuous, also where values of the blend tie,
        and keeps their order. The blend's minimum maps to the lowest target value, so dry areas
        stay dry, and no value goes beyond the target's minimum and maximum. A NaN or infinity
        in the frames or flows makes most of its image NaN.
    quantiles : int
        Number of quantiles of the probability matching, at least 2, at probabilities
        ``k / (quantiles - 1)``. ``H * W`` comes close to rank matching.

    Returns
    -------
    Tensor
        Frames of shape ``(B, T, C, H, W)``, one per time.
    """
    check_frames(frame0, frame1)
    check_flow(flow, frame0)
    if backward_flow is not None:
        check_flow(backward_flow, frame0, name="backward_flow")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise ValueError(f"quantiles must be an integer of at least 2, got {quantiles!r}")
    times = _times(t, frame0)
    batch, steps = frame0.shape[0], times.shape[0]
    t5 = times.view(1, steps, 1, 1, 1)

    def per_time(x: Tensor, scale: Tensor | None = None) -> Tensor:
        """``(B, ...)`` to ``(B * T, ...)``, multiplied by ``scale`` of shape ``(1, T, 1, 1, 1)``."""
        x = x.unsqueeze(1)
        return (x.expand(-1, steps, -1, -1, -1) if scale is None else x * scale).flatten(0, 1)

    flow0 = per_time(flow, -t5)
    flow1 = per_time(flow, 1 - t5) if backward_flow is None else per_time(backward_flow, t5 - 1)
    warped0 = warp(per_time(frame0), flow0).unflatten(0, (batch, steps))
    warped1 = warp(per_time(frame1), flow1).unflatten(0, (batch, steps))
    out = (1 - t5) * warped0 + t5 * warped1
    if match_distribution:
        out = _match_distribution(out, frame0, frame1, times, quantiles)
    # warp at zero displacement is off by grid_sample's rounding; return the frames themselves at
    # the end times, keeping the gradients of the blend (continuous in t, so the same in the limit)
    exact = torch.where(t5 == 0, frame0.unsqueeze(1), frame1.unsqueeze(1))
    return torch.where((t5 == 0) | (t5 == 1), exact.detach() + (out - out.detach()), out)
