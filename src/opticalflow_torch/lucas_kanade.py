"""Dense pyramidal Lucas-Kanade optical flow."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from typing import Literal

import torch
from torch import Tensor, nn

from ._ops import (
    as_scalar,
    check_flow,
    check_frames,
    gaussian_kernel1d,
    no_autocast,
    pyramid_level,
    pyramid_levels,
    resize,
    separable_filter,
)
from .warp import warp

_WINDOWS = ("box", "gaussian")
_LEARNABLE = ("window_sigma", "eps")
_BLUR = (0.25, 0.5, 0.25)  # the blur of the full-resolution pyramid level, applied to the coarser ones too
_RESIDUAL_DAMPING = 1 / 16  # of the unexplained window-averaged squared residual, added to eps; see _update


# ----------------------------------------------------------------------------------------------
# Argument checks
# ----------------------------------------------------------------------------------------------


def _check_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _check_number(value: object, name: str, *, tensor: bool, positive: bool = False) -> None:
    """Check a finite real number (positive if ``positive``), or a 1-element floating tensor."""
    if tensor and isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got a tensor of shape {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        kind = "a float or a tensor" if tensor else "a float"
        raise TypeError(f"{name} must be {kind}, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_options(
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    window: str,
    window_sigma: float | Tensor | None,
    eps: float | Tensor,
    data_range: float,
) -> tuple[float, int, int, int]:
    """Validate the options of :func:`lucas_kanade`; returns ``pyr_scale, levels, winsize, iterations``."""
    _check_number(pyr_scale, "pyr_scale", tensor=False)
    if not 0 < pyr_scale < 1:
        raise ValueError(f"pyr_scale must be in (0, 1), got {pyr_scale}")
    levels = _check_int(levels, "levels", 0)
    winsize = _check_int(winsize, "winsize", 1)
    iterations = _check_int(iterations, "iterations", 1)
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    if window_sigma is not None:
        if window != "gaussian":
            raise ValueError("window_sigma is only valid with window='gaussian'")
        _check_number(window_sigma, "window_sigma", tensor=True, positive=True)
    _check_number(eps, "eps", tensor=True, positive=True)
    _check_number(data_range, "data_range", tensor=False, positive=True)
    return float(pyr_scale), levels, winsize, iterations


# ----------------------------------------------------------------------------------------------
# Steps of the algorithm
# ----------------------------------------------------------------------------------------------


def _window_kernel(window: str, winsize: int, window_sigma: float | Tensor | None, like: Tensor) -> Tensor:
    """1-D averaging kernel of length ``2 * (winsize // 2) + 1``, summing to 1."""
    radius = winsize // 2
    if radius == 0:
        return torch.ones(1, dtype=like.dtype, device=like.device)
    if window == "box":
        end = 0.5 if winsize % 2 == 0 else 1.0  # an even box covers half of its outermost pixels
        return torch.tensor([end] + [1.0] * (2 * radius - 1) + [end], dtype=like.dtype, device=like.device) / winsize
    sigma = 0.3 * radius if window_sigma is None else window_sigma
    return gaussian_kernel1d(as_scalar(sigma, like, "window_sigma"), radius)


def _gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    """Central differences along x and y, replicating the border pixels."""
    # Measured on translations, rotations, zooms and a flow step: Sobel, Scharr and 5-tap
    # derivatives were within 3% of central differences. Gradients of frame0 (as in OpenCV's
    # sparse LK) did best where the flow leaves the frame; those of the warped frame1, alone or
    # averaged with frame0's, let the flow there run off by pixels.
    # Products of differences of padded slices fail to compile (torch 2.15 Inductor).
    kernel = torch.tensor([-0.5, 0.0, 0.5], dtype=image.dtype, device=image.device)
    return separable_filter(image, kernel, None, "replicate"), separable_filter(image, None, kernel, "replicate")


def _inside(flow: Tensor) -> Tensor:
    """1 where ``p + flow`` is inside the frame, falling linearly to 0 one pixel outside; ``(B, 1, H, W)``."""
    height, width = flow.shape[-2:]
    x = torch.arange(width, dtype=flow.dtype, device=flow.device) + flow[:, 0]
    y = torch.arange(height, dtype=flow.dtype, device=flow.device).unsqueeze(-1) + flow[:, 1]
    inside_x = torch.minimum(x + 1, width - x).clamp(0, 1)
    inside_y = torch.minimum(y + 1, height - y).clamp(0, 1)
    return (inside_x * inside_y).unsqueeze(1)


def _inverse(g11: Tensor, g12: Tensor, g22: Tensor, mu: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(a11, a12, a22, det)`` with ``(G + mu I) ** -1 = [[a22, -a12], [-a12, a11]] / det``."""
    # Normalised by the trace, so that flat windows (G = 0) divide by mu and not by mu ** 2, which
    # underflows in float32 for eps below 1e-19 and makes the backward overflow.
    trace = g11 + g22 + 2 * mu
    g11, g12, g22, mu = g11 / trace, g12 / trace, g22 / trace, mu / trace
    # det(G) >= 0 exactly; the clamp only removes rounding that could flip the sign of the update
    det = ((g11 * g22 - g12 * g12).clamp(min=0) + mu * (g11 + g22 + mu)) * trace
    return g11 + mu, g12, g22 + mu, det


def _update(averaged: Tensor, eps: Tensor) -> Tensor:
    """``delta`` with ``(G + mu I) delta = -b``, from the window-averaged ``(g11, g12, g22, b1, b2, r2)``."""
    g11, g12, g22, b1, b2, r2 = averaged.unbind(1)
    # With mu = eps, the update reaches r / (2 sqrt(eps)) where the texture of frame0 is about
    # sqrt(eps), and its derivative r / eps where frame0 is flat. The squared residual that the
    # eps-damped update leaves, r2 - b.(G + eps I)^-1 b >= 0, is about r2 there but vanishes where a
    # motion explains the residual. Since |b| <= sqrt(l * r2) along each eigenvector of G (eigenvalue
    # l, Cauchy-Schwarz), adding it to mu bounds such updates by 1 / (2 sqrt(_RESIDUAL_DAMPING)).
    # Adding r2 itself also held back the weak direction of nearly rank-one G (smooth textures): 3-8
    # times the error on 1-2.5 px motions without a pyramid.
    a11, a12, a22, det = _inverse(g11, g12, g22, eps)
    explained = (a22 * b1 * b1 - 2 * a12 * b1 * b2 + a11 * b2 * b2) / det
    mu = eps + _RESIDUAL_DAMPING * (r2 - explained).clamp(min=0)
    a11, a12, a22, det = _inverse(g11, g12, g22, mu)
    return torch.stack(((a12 * b2 - a22 * b1) / det, (a12 * b1 - a11 * b2) / det), dim=1)


def _min_eigenvalue(averaged: Tensor) -> Tensor:
    g11, g12, g22 = averaged[:, 0], averaged[:, 1], averaged[:, 2]
    # vector_norm has a zero gradient at zero, where sqrt's would be infinite
    radius = torch.linalg.vector_norm(torch.stack(((g11 - g22) * 0.5, g12)), dim=0)
    return (g11 + g22) * 0.5 - radius


def _resize_flow(flow: Tensor, size: tuple[int, int], interpolation: Literal["linear", "area"]) -> Tensor:
    """Resize a flow with :func:`~opticalflow_torch._ops.resize` and rescale each component to the new grid."""
    height, width = flow.shape[-2:]
    out = resize(flow, size, interpolation)
    return torch.cat((out[:, :1] * (size[1] / width), out[:, 1:] * (size[0] / height)), dim=1)


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------


@no_autocast
def lucas_kanade(
    frame0: Tensor,
    frame1: Tensor,
    *,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 21,
    iterations: int = 5,
    window: Literal["box", "gaussian"] = "gaussian",
    window_sigma: float | Tensor | None = None,
    eps: float | Tensor = 1e-3,
    data_range: float = 255.0,
    initial_flow: Tensor | None = None,
    return_confidence: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Dense optical flow from ``frame0`` to ``frame1`` with Lucas-Kanade.

    At every pixel, solves the Lucas-Kanade equations averaged over a window around it,
    iterating with warping and going from coarse to fine pyramid levels. Spatial gradients are
    central differences of ``frame0``; pixels displaced outside ``frame1`` drop out of the
    equations. Before each iteration the flow is also averaged over the window, which, unlike
    the constant flow per window of classic Lucas-Kanade, leaves rotations and zooms unbiased.
    The pyramid is that of :func:`~opticalflow_torch.farneback`, with the coarser levels also
    blurred by ``[1, 2, 1] / 4`` as the full-resolution one is. Besides ``eps``, each update is
    damped by 1/16 of the window-averaged squared residual that no displacement explains, which
    keeps updates below about 2 pixels of the level, and gradients moderate, where ``frame0`` is
    flat but ``frame1`` is not. Differentiable with respect to both frames, ``initial_flow``,
    ``window_sigma`` and ``eps``. The defaults assume frames spanning 0-255; give ``data_range``
    for others, e.g. ``data_range=1.0`` for frames in [0, 1]. Frames with several channels sum
    their per-channel equations. float16 and bfloat16 frames are processed in float32, and
    float32 frames stay in float32 inside :func:`torch.autocast`.

    Parameters
    ----------
    frame0, frame1 : Tensor
        Frames, shape ``(B, C, H, W)``, floating point.
    pyr_scale : float
        Size ratio between consecutive pyramid levels, below 1.
    levels : int
        Number of coarser levels added on top of the full-resolution frames; 0 means no
        pyramid. Same convention as :func:`~opticalflow_torch.farneback`, including the
        skipping of levels smaller than 32 pixels.
    winsize : int
        Size of the window over which each pixel's equations are averaged. An even size gives
        half weight to the outermost rows and columns.
    iterations : int
        Iterations at each pyramid level, at least 1.
    window : {"box", "gaussian"}
        Weighting of the window.
    window_sigma : float or Tensor, optional
        Sigma of the Gaussian window, by default ``0.3 * (winsize // 2)``. Only valid with
        ``window="gaussian"``.
    eps : float or Tensor
        Positive, added to the diagonal of each window-averaged structure tensor (mean squared
        intensity gradients) of frames spanning 0-255, so it does not depend on ``winsize``; for
        frames spanning ``data_range``, ``eps * (data_range / 255) ** 2``. It damps updates where
        there is no texture, a smooth version of OpenCV's minimum-eigenvalue cut. Floored at
        about 1e-19 in float32 and 1e-154 in float64.
    data_range : float
        Span of the frame values, positive: 255 for 0-255 frames, 1.0 for frames in [0, 1].
        Frames scaled together with it give the same flow. Too large a value holds the flow
        back where the texture is weak.
    initial_flow : Tensor, optional
        Starting estimate, shape ``(B, 2, H, W)``.
    return_confidence : bool
        Also return the smallest eigenvalue of each pixel's window-averaged structure tensor
        (without ``eps``, pixels displaced outside ``frame1`` counting as zero) at full
        resolution in the last iteration, shape ``(B, H, W)``. It scales with the square of the
        frame values, not with ``data_range``.

    Returns
    -------
    Tensor or tuple of Tensor
        Flow of shape ``(B, 2, H, W)`` in pixels, and the confidence if requested.
    """
    check_frames(frame0, frame1)
    pyr_scale, levels, winsize, iterations = _check_options(
        pyr_scale, levels, winsize, iterations, window, window_sigma, eps, data_range
    )
    if initial_flow is not None:
        check_flow(initial_flow, frame0, "initial_flow")
    dtype = frame0.dtype
    compute = torch.promote_types(dtype, torch.float32)
    batch, _, height, width = frame0.shape
    # in units of 0-255 frames, where eps applies: the same as eps * (data_range / 255) ** 2 on the
    # frames as given, with the floor below at the same place for every data_range
    scale = 255 / data_range
    frames = torch.cat((frame0, frame1)).to(compute) * scale
    kernel = _window_kernel(window, winsize, window_sigma, frames)
    # below this, 1 / eps on flat windows overflows in the backward
    eps = as_scalar(eps, frames, "eps").clamp(min=torch.finfo(compute).tiny ** 0.5)
    blur = torch.tensor(_BLUR, dtype=compute, device=frames.device)

    pyramid = pyramid_levels(height, width, pyr_scale, levels)
    flow = None if initial_flow is None else initial_flow.to(compute)
    for k in range(len(pyramid) - 1, -1, -1):
        size = (pyramid[k].height, pyramid[k].width)
        if k < len(pyramid) - 1:
            flow = _resize_flow(flow, size, "linear")
        elif flow is None:
            flow = frames.new_zeros(batch, 2, *size)
        else:
            flow = _resize_flow(flow, size, "area")
        images = pyramid_level(frames, pyramid[k])
        if k > 0:
            # The coarser levels are only lightly blurred before resampling, enough for Farneback's
            # polynomial fits but not for differences: their near-Nyquist content cut the updates
            # to fractions of a pixel, and a 20 px motion was not found in 5 iterations.
            images = separable_filter(images, blur, blur)
        image0, image1 = images.split(batch)
        gx, gy = _gradients(image0)
        structure = torch.stack(((gx * gx).sum(1), (gx * gy).sum(1), (gy * gy).sum(1)), dim=1)
        for _ in range(iterations):
            # Without this, per-pixel updates let neighbouring estimates drift apart over the
            # iterations. Referring each window to its centre's flow instead (classic LK, also
            # Farneback) is stable but biased on affine flows: 3-14 times the error on 3 degree
            # rotations and 4% zooms, which this keeps; it costs 10-40% on translations.
            flow = separable_filter(flow, kernel, kernel)
            residual = warp(image1, flow) - image0
            products = (gx * residual, gy * residual, residual * residual)
            terms = torch.cat((structure, *(p.sum(1, True) for p in products)), dim=1)
            averaged = separable_filter(terms * _inside(flow), kernel, kernel)
            flow = flow + _update(averaged, eps)
    if return_confidence:
        return flow.to(dtype), (_min_eigenvalue(averaged) / scale**2).to(dtype)
    return flow.to(dtype)


def _learnable_names(learnable: bool | Iterable[str], window: str, winsize: int) -> set[str]:
    has_window_sigma = window == "gaussian" and winsize >= 2
    if isinstance(learnable, bool):
        return {n for n in _LEARNABLE if n != "window_sigma" or has_window_sigma} if learnable else set()
    if isinstance(learnable, str):
        learnable = [learnable]
    try:
        names = list(learnable)
    except TypeError:
        raise TypeError(f"learnable must be a bool or an iterable of str, got {type(learnable).__name__}") from None
    for name in names:
        if name not in _LEARNABLE:
            raise ValueError(f"learnable parameters must be among {_LEARNABLE}, got {name!r}")
    if "window_sigma" in names and not has_window_sigma:
        raise ValueError("window_sigma can only be learned with window='gaussian' and winsize >= 2")
    return set(names)


class LucasKanade(nn.Module):
    """Module form of :func:`lucas_kanade`, with optionally learnable parameters.

    Parameters are those of :func:`lucas_kanade` except the frames, ``initial_flow`` and
    ``return_confidence``, which go to :meth:`forward`.

    Parameters
    ----------
    learnable : bool or iterable of str
        Parameters to register as :class:`torch.nn.Parameter`, among ``"window_sigma"``
        (Gaussian window only) and ``"eps"``; ``True`` means all that apply. They are stored
        as logarithms so they stay positive. ``eps`` stays in units of 0-255 frames, so a
        learned value carries over to another ``data_range``, which is not learnable.
    """

    def __init__(
        self,
        *,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 21,
        iterations: int = 5,
        window: Literal["box", "gaussian"] = "gaussian",
        window_sigma: float | None = None,
        eps: float = 1e-3,
        data_range: float = 255.0,
        learnable: bool | Iterable[str] = False,
    ) -> None:
        super().__init__()
        for name, value in (("window_sigma", window_sigma), ("eps", eps)):
            if value is not None:
                _check_number(value, name, tensor=False)
        self.pyr_scale, self.levels, self.winsize, self.iterations = _check_options(
            pyr_scale, levels, winsize, iterations, window, window_sigma, eps, data_range
        )
        self.window = window
        self.data_range = float(data_range)
        names = _learnable_names(learnable, window, self.winsize)
        if window_sigma is None and "window_sigma" in names:
            window_sigma = 0.3 * (self.winsize // 2)
        self._fixed: dict[str, float | None] = {}
        for name, value in (("window_sigma", window_sigma), ("eps", eps)):
            if name in names:
                self.register_parameter(f"log_{name}", nn.Parameter(torch.tensor(math.log(value))))
            else:
                self._fixed[name] = None if value is None else float(value)

    def _value(self, name: str) -> float | Tensor | None:
        if name in self._fixed:
            return self._fixed[name]
        return getattr(self, f"log_{name}").exp()

    @property
    def window_sigma(self) -> float | Tensor | None:
        """Current ``window_sigma``, a tensor if learnable."""
        return self._value("window_sigma")

    @property
    def eps(self) -> float | Tensor:
        """Current ``eps``, a tensor if learnable."""
        return self._value("eps")

    @no_autocast
    def forward(
        self,
        frame0: Tensor,
        frame1: Tensor,
        initial_flow: Tensor | None = None,
        return_confidence: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Flow from ``frame0`` to ``frame1``; see :func:`lucas_kanade`."""
        return lucas_kanade(
            frame0,
            frame1,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            window=self.window,
            window_sigma=self.window_sigma,
            eps=self.eps,
            data_range=self.data_range,
            initial_flow=initial_flow,
            return_confidence=return_confidence,
        )

    def extra_repr(self) -> str:
        fields = [
            f"pyr_scale={self.pyr_scale}",
            f"levels={self.levels}",
            f"winsize={self.winsize}",
            f"iterations={self.iterations}",
            f"window={self.window!r}",
        ]
        fields += [f"{name}={value}" for name, value in self._fixed.items() if value is not None]
        fields.append(f"data_range={self.data_range}")
        learnable = [name for name in _LEARNABLE if name not in self._fixed]
        if learnable:
            fields.append(f"learnable={learnable}")
        return ", ".join(fields)
